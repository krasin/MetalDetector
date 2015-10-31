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

    var metalDevice: MTLDevice?
    var metalLib: MTLLibrary?
    var commandQueue: MTLCommandQueue?
    var textureCache : Unmanaged<CVMetalTextureCacheRef>?
    var input: MTLTexture?
    var output : MTLTexture?
    var outputBGRA : MTLTexture?

    // Shaders
    var cropAndRotateState: MTLComputePipelineState?
    var crop352x288to224State: MTLComputePipelineState?
    var float2BGRAState : MTLComputePipelineState?

    var engine: Engine?
    var net : Net?

    @IBOutlet weak var infoLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()

        if NSProcessInfo.processInfo().environment["SAMOFLY_UNIT_TESTS"] == nil {
            // Initialize Metal
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

    private func loadKernelState(kernelName: String) -> MTLComputePipelineState? {
        let f = metalLib!.newFunctionWithName(kernelName)
        if f == nil {
            print("Could not load \(kernelName) from the library")
            exit(1)
        }
        do {
            return try metalDevice!.newComputePipelineStateWithFunction(f!)
        } catch {
            print("Could not create pipeline state for \(kernelName)")
            exit(1)
        }
    }

    private func initMetal() {
        engine = Engine()
        net = Net(engine: engine!, config: GoogLeNetConfig(),
            threadsPerThreadgroup: GoogLeNetProfile.GetThreadsPerThreadgroup())
        metalDevice = MTLCreateSystemDefaultDevice()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice!, nil, &textureCache)
        metalLib = metalDevice!.newDefaultLibrary()!
        commandQueue = metalDevice!.newCommandQueue()

        cropAndRotateState = loadKernelState("cropAndRotate")
        crop352x288to224State = loadKernelState("crop352x288to224")
        float2BGRAState = loadKernelState("float2BGRA")

        // init output texture
        let outputDesc = MTLTextureDescriptor()
        outputDesc.textureType = MTLTextureType.Type2DArray
        outputDesc.height = windowSize
        outputDesc.width = windowSize
        outputDesc.pixelFormat = MTLPixelFormat.R16Float
        outputDesc.arrayLength = 3
        // outputDesc.mipmapLevelCount = 1
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
        /*if self.frameCount % 100 == 0 {
            let count = self.frameCount
            dispatch_async(dispatch_get_main_queue(), {
                self.infoLabel.text = "Frame #\(count)"
            })
            print("Frame #\(self.frameCount)")
        }*/
        self.frameCount++
        if self.frameCount % 100 != 1 {
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

        // Crop
        /*if true {
            let commandBuffer = commandQueue!.commandBuffer()
            let commandEncoder = commandBuffer.computeCommandEncoder()
            //let sampleDesc = MTLSamplerDescriptor()
            //sampleDesc.normalizedCoordinates = true
            //sampleDesc.minFilter = MTLSamplerMinMagFilter.Linear
            //sampleDesc.magFilter = MTLSamplerMinMagFilter.Linear
            //sampleDesc.maxAnisotropy = 16
            //let sampleState = metalDevice!.newSamplerStateWithDescriptor(sampleDesc)
            commandEncoder.setComputePipelineState(crop352x288to224State!)
            commandEncoder.setTexture(input, atIndex: 0)
            commandEncoder.setTexture(output, atIndex: 1)
            //commandEncoder.setSamplerState(sampleState, atIndex: 0)
            let threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
            let threadgroupsPerGrid = MTLSizeMake(windowSize / threadsPerThreadgroup.width,
                windowSize / threadsPerThreadgroup.height, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            commandEncoder.endEncoding()
            commandBuffer.commit();
            commandBuffer.waitUntilCompleted()
        }*/
        let ans = net!.forward(input!)
        let (idx, prob) = argMax(ans)
        let label = "\(net!.labels[idx]) - \(prob*100)%"
        let workTime = NSDate().timeIntervalSinceDate(startTime)
        print("net.forward is done within \(workTime) sec")

        print("GoogLeNet: \(label)")
        dispatch_async(dispatch_get_main_queue(), {
            self.infoLabel.text = label
        })

        /*if true {
            let commandBuffer = commandQueue!.commandBuffer()
            let commandEncoder = commandBuffer.computeCommandEncoder()
            commandEncoder.setComputePipelineState(float2BGRAState!)
            commandEncoder.setTexture(net!.blobs["data"]!, atIndex: 0)
            commandEncoder.setTexture(outputBGRA, atIndex: 1)
            let threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
            let threadgroupsPerGrid = MTLSizeMake(windowSize / threadsPerThreadgroup.width,
                windowSize / threadsPerThreadgroup.height, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            commandEncoder.endEncoding()
            commandBuffer.commit();
            commandBuffer.waitUntilCompleted()
        }*/


        // Copy data from the output texture.
        /*var imageBuf = Array<UInt8>(count: windowSize * windowSize * 4, repeatedValue: 0)
        outputBGRA!.getBytes(&imageBuf, bytesPerRow: windowSize*4,
            fromRegion: MTLRegionMake2D(0, 0, windowSize, windowSize), mipmapLevel: 0)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGBitmapContextCreate(&imageBuf, windowSize, windowSize, 8, 4*windowSize, colorSpace,
            CGImageAlphaInfo.PremultipliedFirst.rawValue | CGBitmapInfo.ByteOrder32Little.rawValue)
        if context == nil {
            print("Could not create context from image buf")
            exit(1)
        }
        let cgImg = CGBitmapContextCreateImage(context)
        if cgImg == nil {
            print("Could not create CGImage from context")
            exit(1)
        }
        let uiImg = UIImage(CGImage:cgImg!, scale: 1, orientation: UIImageOrientation.Up)
        UIImageWriteToSavedPhotosAlbum(uiImg, nil, nil, nil)*/
        // DO NOT FORGET to release cgImg and uiImg.
    }
}
