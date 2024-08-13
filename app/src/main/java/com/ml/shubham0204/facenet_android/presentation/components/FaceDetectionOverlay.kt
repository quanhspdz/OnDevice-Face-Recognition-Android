package com.ml.shubham0204.facenet_android.presentation.components

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Size
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.FrameLayout
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.toRectF
import androidx.core.view.doOnLayout
import androidx.lifecycle.LifecycleOwner
import com.ml.shubham0204.facenet_android.presentation.screens.detect_screen.DetectScreenViewModel
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Suppress("DEPRECATION")
@SuppressLint("ViewConstructor")
@ExperimentalGetImage
class FaceDetectionOverlay(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context,
    private val viewModel: DetectScreenViewModel
) : FrameLayout(context) {

    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0

    private var imageTransform: Matrix = Matrix()
    private var boundingBoxTransform: Matrix = Matrix()
    private var isImageTransformedInitialized = false
    private var isBoundingBoxTransformedInitialized = false

    private lateinit var frameBitmap: Bitmap
    private var isProcessing = false
    private var cameraFacing: Int = CameraSelector.LENS_FACING_FRONT
    private lateinit var boundingBoxOverlay: BoundingBoxOverlay
    private lateinit var previewView: PreviewView

    var predictions: Array<Prediction> = arrayOf()

    init {
        initializeCamera(cameraFacing)
        doOnLayout {
            overlayHeight = it.measuredHeight
            overlayWidth = it.measuredWidth
        }
    }

    fun initializeCamera(cameraFacing: Int) {
        this.cameraFacing = cameraFacing
        this.isImageTransformedInitialized = false
        this.isBoundingBoxTransformedInitialized = false
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val previewView = PreviewView(context)
        val executor = ContextCompat.getMainExecutor(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder()
                    .setTargetResolution(Size(480, 640)) // Set the desired resolution here
                    .build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                val cameraSelector =
                    CameraSelector.Builder().requireLensFacing(cameraFacing).build()
                val frameAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                frameAnalyzer.setAnalyzer(Executors.newSingleThreadExecutor(), analyzer)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    frameAnalyzer
                )
            },
            executor
        )

        if (childCount == 2) {
            removeView(this.previewView)
            removeView(this.boundingBoxOverlay)
        }

        this.previewView = previewView
        addView(this.previewView)

        val boundingBoxOverlayParams =
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        this.boundingBoxOverlay = BoundingBoxOverlay(context)
        this.boundingBoxOverlay.setWillNotDraw(false)
        this.boundingBoxOverlay.setZOrderOnTop(true)
        addView(this.boundingBoxOverlay, boundingBoxOverlayParams)
    }

    private val analyzer =
        ImageAnalysis.Analyzer { image ->
            if (isProcessing) {
                image.close()
                return@Analyzer
            }
            isProcessing = true

            // Transform android.net.Image to Bitmap
            val imageProxy = image
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height
            frameBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888)
            frameBitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

            // Configure frameHeight and frameWidth for output2overlay transformation matrix
            if (!isImageTransformedInitialized) {
                imageTransform =
                    Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
                isImageTransformedInitialized = true
            }
            frameBitmap = Bitmap.createBitmap(
                frameBitmap,
                0,
                0,
                frameBitmap.width,
                frameBitmap.height,
                imageTransform,
                false
            )

            if (!isBoundingBoxTransformedInitialized) {
                boundingBoxTransform = Matrix().apply {
                    setScale(
                        overlayWidth / frameBitmap.width.toFloat(),
                        overlayHeight / frameBitmap.height.toFloat()
                    )
                    if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
                        // Mirror the bounding box coordinates for the front-facing camera
                        postScale(
                            -1f,
                            1f,
                            overlayWidth.toFloat() / 2.0f,
                            overlayHeight.toFloat() / 2.0f
                        )
                    }
                }
                isBoundingBoxTransformedInitialized = true
            }

            CoroutineScope(Dispatchers.Default).launch {
                val predictions = ArrayList<Prediction>()
                val (metrics, results) = viewModel.imageVectorUseCase.getNearestPersonName(
                    frameBitmap
                )
                results.forEach { (name, boundingBox) ->
                    val box = boundingBox.toRectF()
                    var personName = name
                    if (viewModel.getNumPeople().toInt() == 0) {
                        personName = ""
                    }
                    boundingBoxTransform.mapRect(box)
                    predictions.add(Prediction(box, personName))
                }
                withContext(Dispatchers.Main) {
                    viewModel.faceDetectionMetricsState.value = metrics
                    this@FaceDetectionOverlay.predictions = predictions.toTypedArray()
                    boundingBoxOverlay.invalidate()
                    isProcessing = false
                }
            }
            image.close()
        }

    data class Prediction(var bbox: RectF, var label: String, var maskLabel: String = "")

    inner class BoundingBoxOverlay(context: Context) :
        SurfaceView(context), SurfaceHolder.Callback {

        private val boxPaint =
            Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 8.0f
            }
        private val textPaint =
            Paint().apply {
                strokeWidth = 4.0f
                textSize = 50f
                color = Color.GREEN
            }

        override fun surfaceCreated(holder: SurfaceHolder) {}

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

        override fun surfaceDestroyed(holder: SurfaceHolder) {}

        override fun onDraw(canvas: Canvas) {
            predictions.forEach {
                canvas.drawRoundRect(it.bbox, 8f, 8f, boxPaint)
                canvas.drawText(it.label, it.bbox.left, it.bbox.top - 10f, textPaint)
            }
        }
    }
}
