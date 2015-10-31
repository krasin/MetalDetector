//
//  ViewController.swift
//  MetalDetector
//
//  Created by Ivan Krasin on 9/27/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import AVFoundation
import Darwin
import Metal
import MetalKit
import MetalPerformanceShaders
import UIKit


class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    // Size of the input image for the neural net.
    let windowSize = 224

    var previewLayer: AVCaptureVideoPreviewLayer?
    var frameCount = 0
    var running: Bool = false

    var metalDevice: MTLDevice?
    var textureCache : Unmanaged<CVMetalTextureCacheRef>?
    var input: MTLTexture?
    var output : MTLTexture?
    var outputBGRA : MTLTexture?

    var engine: Engine?
    var net : Net?

    @IBOutlet weak var infoLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()

        if NSProcessInfo.processInfo().environment["SAMOFLY_UNIT_TESTS"] == nil {
            initMetal()
        }

        // Initialize UI
        infoLabel.layer.cornerRadius=8.0;
        infoLabel.clipsToBounds = true

        // Initialize video recorder
        let session = AVCaptureSession()
        session.sessionPreset = AVCaptureSessionPreset352x288

        // Prepare input
        do {
            let camera = AVCaptureDevice.defaultDeviceWithMediaType(AVMediaTypeVideo)
            let input = try AVCaptureDeviceInput(device: camera)
            session.addInput(input)
        } catch {
            print("Can't get the camera input")
            exit(1)
        }

        // Prepare output
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        out.alwaysDiscardsLateVideoFrames = true
        let queue = dispatch_queue_create("input frames queue", DISPATCH_QUEUE_SERIAL)
        out.setSampleBufferDelegate(self, queue:queue)
        if !session.canAddOutput(out) {
            print("Can't add video preview output")
            exit(1)
        }
        session.addOutput(out)

        // Create and register the video preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer!.videoGravity = AVLayerVideoGravityResizeAspectFill
        previewLayer!.zPosition = -1
        view.layer.addSublayer(previewLayer!)

        if NSProcessInfo.processInfo().environment["SAMOFLY_UNIT_TESTS"] == nil {
            session.startRunning()
        }
    }

    override func viewDidLayoutSubviews() {
        previewLayer!.frame = view.bounds
        let connection = previewLayer!.connection
        // Make sure the preview layer is always correctly oriented.
        if connection.supportsVideoOrientation {
            let statusBarOrientation = UIApplication.sharedApplication().statusBarOrientation
            switch statusBarOrientation {
            case UIInterfaceOrientation.Portrait:
                connection.videoOrientation = AVCaptureVideoOrientation.Portrait
            case UIInterfaceOrientation.PortraitUpsideDown:
                connection.videoOrientation = AVCaptureVideoOrientation.PortraitUpsideDown
            case UIInterfaceOrientation.LandscapeLeft:
                connection.videoOrientation = AVCaptureVideoOrientation.LandscapeLeft
            case UIInterfaceOrientation.LandscapeRight:
                connection.videoOrientation = AVCaptureVideoOrientation.LandscapeRight
            default:
                connection.videoOrientation = AVCaptureVideoOrientation.Portrait
            }
        }
    }

    private func initMetal() {
        engine = Engine()
        net = Net(engine: engine!, config: GoogLeNetConfig(),
            threadsPerThreadgroup: GoogLeNetProfile.GetThreadsPerThreadgroup())
        metalDevice = MTLCreateSystemDefaultDevice()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice!, nil, &textureCache)

        // init output texture
        let outputDesc = MTLTextureDescriptor()
        outputDesc.textureType = MTLTextureType.Type2DArray
        outputDesc.height = windowSize
        outputDesc.width = windowSize
        outputDesc.pixelFormat = MTLPixelFormat.R16Float
        outputDesc.arrayLength = 3
        output = metalDevice!.newTextureWithDescriptor(outputDesc)

        let outputDescBGRA = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat(
            MTLPixelFormat.BGRA8Unorm, width:windowSize, height:windowSize, mipmapped:false)
        outputBGRA = metalDevice!.newTextureWithDescriptor(outputDescBGRA)
    }

    func argMax(arr : [Float]) -> (Int, Float) {
        var maxv : Float = arr[0]
        var idx : Int = 0
        for i in 0...arr.count-1 {
            if arr[i] > maxv {
                maxv = arr[i]
                idx = i
            }
        }
        return (idx, maxv)
    }

    func captureOutput(captureOutput: AVCaptureOutput!,
            didOutputSampleBuffer sampleBuffer: CMSampleBuffer!,
            fromConnection connection: AVCaptureConnection!) {
        self.frameCount++
        if self.frameCount < 50 {
            return
        }
        var ok = false
        objc_sync_enter(self.running)
        if !self.running {
            ok = true
            self.running = true
        }
        objc_sync_exit(self.running)
        if !ok {
            return
        }

        let buf = CMSampleBufferGetImageBuffer(sampleBuffer)
        var texture : Unmanaged<CVMetalTextureRef>?
        let w = CVPixelBufferGetWidthOfPlane(buf!, 0);
        let h = CVPixelBufferGetHeightOfPlane(buf!, 0);
        print("w=\(w), h=\(h)")

        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                textureCache!.takeUnretainedValue(),
                buf!, nil, MTLPixelFormat.BGRA8Unorm, w, h, 0, &texture)
        if texture == nil {
            print("Failed to create a texture from image")
            exit(1)
        }
        input = CVMetalTextureGetTexture((texture?.takeUnretainedValue())!)
        texture!.release()

        // Run Metal shaders on the input and fill the output
        let startTime = NSDate()

        let ans = net!.forward(input!)
        let (idx, prob) = argMax(ans)
        let label = "\(net!.labels[idx]) - \(prob*100)%"
        let workTime = NSDate().timeIntervalSinceDate(startTime)
        print("net.forward is done within \(workTime) sec")
        objc_sync_enter(self.running)
        self.running = false
        objc_sync_exit(self.running)

        print("GoogLeNet: \(label)")
        dispatch_async(dispatch_get_main_queue(), {
            self.infoLabel.text = label
        })
    }
}
