package com.ml.shubham0204.facenet_android.presentation.components

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
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
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.shubham0204.facenet_android.presentation.viewmodels.DetectScreenViewModel
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@SuppressLint("ViewConstructor")
@ExperimentalGetImage
class FaceDetectionOverlay(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context,
    private val viewModel: DetectScreenViewModel
) : FrameLayout(context) {

    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0
    private val realTimeOpts =
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.4f)
            .build()
    private val detector = FaceDetection.getClient(realTimeOpts)

    private var imageTransform: Matrix = Matrix()
    private var boundingBoxTransform: Matrix = Matrix()
    private var isImageTransformedInitialized = false
    private var isBoundingBoxTransformedInitialized = false

    private var isProcessing = false
    private var cameraFacing: Int = CameraSelector.LENS_FACING_BACK
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
                val preview =
                    Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                val cameraSelector =
                    CameraSelector.Builder().requireLensFacing(cameraFacing).build()
                val frameAnalyzer =
                    ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
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

            var frameBitmap =
                Bitmap.createBitmap(
                    image.image!!.width,
                    image.image!!.height,
                    Bitmap.Config.ARGB_8888
                )
            frameBitmap.copyPixelsFromBuffer(image.planes[0].buffer)

            // Configure frameHeight and frameWidth for output2overlay transformation matrix.
            if (!isImageTransformedInitialized) {
                imageTransform = Matrix()
                imageTransform.apply {
                    postRotate(image.imageInfo.rotationDegrees.toFloat())
                    if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
                        postScale(-1f, 1f, image.width.toFloat(), image.height.toFloat())
                    }
                }
                isImageTransformedInitialized = true
            }

            frameBitmap =
                Bitmap.createBitmap(
                    frameBitmap,
                    0,
                    0,
                    frameBitmap.width,
                    frameBitmap.height,
                    imageTransform,
                    false
                )

            if (!isBoundingBoxTransformedInitialized) {
                boundingBoxTransform = Matrix()
                boundingBoxTransform.apply {
                    postScale(
                        overlayWidth / frameBitmap.width.toFloat(),
                        overlayHeight / frameBitmap.height.toFloat()
                    )
                }
                isBoundingBoxTransformedInitialized = true
            }

            val inputImage = InputImage.fromBitmap(frameBitmap, 0)

            detector
                .process(inputImage)
                .addOnSuccessListener { faces ->
                    CoroutineScope(Dispatchers.Default).launch { showFaces(faces, frameBitmap) }
                }
                .addOnCompleteListener { image.close() }
        }

    private suspend fun showFaces(faces: List<Face>, frameBitmap: Bitmap) {
        withContext(Dispatchers.Default) {
            val predictions = ArrayList<Prediction>()
            faces.forEach {
                val t1 = System.currentTimeMillis()
                val name = viewModel.detectFace(frameBitmap)
                Log.e( "APP" , "Retrieval time: ${System.currentTimeMillis() - t1}")
                val box = it.boundingBox.toRectF()
                boundingBoxTransform.mapRect(box)
                predictions.add(
                    Prediction(
                        box,
                        name
                    )
                )
            }
            withContext(Dispatchers.Main) {
                this@FaceDetectionOverlay.predictions = predictions.toTypedArray()
                boundingBoxOverlay.invalidate()
                isProcessing = false
            }
        }
    }

    data class Prediction(var bbox: RectF, var label: String, var maskLabel: String = "")

    inner class BoundingBoxOverlay(context: Context) :
        SurfaceView(context), SurfaceHolder.Callback {

        private val boxPaint =
            Paint().apply {
                color = Color.parseColor("#4D90caf9")
                style = Paint.Style.FILL
            }
        private val textPaint =
            Paint().apply {
                strokeWidth = 2.0f
                textSize = 32f
                color = Color.WHITE
            }

        override fun surfaceCreated(holder: SurfaceHolder) {}

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

        override fun surfaceDestroyed(holder: SurfaceHolder) {}

        override fun onDraw(canvas: Canvas) {
            predictions.forEach {
                canvas.drawRoundRect(it.bbox, 16f, 16f, boxPaint)
                canvas.drawText(it.label, it.bbox.centerX(), it.bbox.centerY(), textPaint)
            }
        }
    }
}
