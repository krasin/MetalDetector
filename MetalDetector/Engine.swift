//
//  Layers.swift
//  MetalDetector
//
//  Created by Ivan Krasin on 10/8/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import UIKit

public class Engine {
    public var metalDevice: MTLDevice?
    var metalLib: MTLLibrary?
    public var commandQueue: MTLCommandQueue?
    var textureCache : Unmanaged<CVMetalTextureCacheRef>?

    var kernelStates : [String: MTLComputePipelineState?]
    var computeL1State: MTLComputePipelineState?
    var computeL2State: MTLComputePipelineState?
    var computeMaxState: MTLComputePipelineState?
    var sample8x8State : MTLComputePipelineState?

    public init() {
        metalDevice = MTLCreateSystemDefaultDevice()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice!, nil, &textureCache)
        metalLib = metalDevice!.newDefaultLibrary()!
        commandQueue = metalDevice!.newCommandQueue()

        // Load kernels
        kernelStates = [:]
        computeL1State = loadKernelState("computeL1")
        computeL2State = loadKernelState("computeL2")
        computeMaxState = loadKernelState("computeMax")
        sample8x8State = loadKernelState("sample8x8")
    }

    private func loadKernelState(kernelName: String) -> MTLComputePipelineState {
        let state = kernelStates[kernelName]
        if state != nil {
            return state!!
        }
        let f = metalLib!.newFunctionWithName(kernelName)
        if f == nil {
            print("Could not load \(kernelName) from the library")
            exit(1)
        }
        do {
            return try metalDevice!.newComputePipelineStateWithFunction(f!)
        } catch let error as NSError {
            print("Could not create pipeline state for \(kernelName): \(error)")
            exit(1)
        }
    }

    public func UnaryLayer(commandBuffer : MTLCommandBuffer, name : String, weights : MTLBuffer?,
        input : MTLTexture, output : MTLTexture, threadsPerThreadgroup: MTLSize) {
        let state = loadKernelState(name)
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(state)
        commandEncoder.setTexture(input, atIndex: 0)
        commandEncoder.setTexture(output, atIndex: 1)
        if weights != nil {
            commandEncoder.setBuffer(weights!, offset: 0, atIndex: 0)
        }
        let threadgroupsPerGrid = MTLSizeMake(
            ((output.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width),
            (output.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }

    public func PerFilterLayer(commandBuffer : MTLCommandBuffer, name : String,
        weights : MTLBuffer?, numFilters : Int, input : MTLTexture, output : MTLTexture) {
            let state = loadKernelState(name)
            let commandEncoder = commandBuffer.computeCommandEncoder()
            commandEncoder.setComputePipelineState(state)
            commandEncoder.setTexture(input, atIndex: 0)
            commandEncoder.setTexture(output, atIndex: 1)
            if weights != nil {
                commandEncoder.setBuffer(weights!, offset: 0, atIndex: 0)
            }
            let threadsPerThreadgroup = MTLSizeMake(256, 1, 1)
            let threadgroupsPerGrid = MTLSizeMake(
                ((numFilters + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width),
                1, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            commandEncoder.endEncoding()
    }

    func Preprocess(commandBuffer : MTLCommandBuffer, input : MTLTexture, output : MTLTexture) {
        let threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
        UnaryLayer(commandBuffer, name: "preprocess", weights: nil, input: input,
            output: output, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    // L1 computes L1 metric. It requires a 2d float texture array as an input.
    public func L1(texture : MTLTexture) -> Float {
        let commandBuffer = commandQueue!.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(computeL1State!)
        commandEncoder.setTexture(texture, atIndex: 0)
        let resBuf = metalDevice!.newBufferWithLength(4 * texture.height, options: .StorageModeShared)
        commandEncoder.setBuffer(resBuf, offset: 0, atIndex: 0)
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let threadgroupsPerGrid = MTLSizeMake(
            (texture.height + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()
        let resArr = UnsafeMutablePointer<Float>(resBuf.contents())
        var res : Float = 0;
        for i in 0...texture.height - 1 {
            res += resArr[i]
        }
        return res
    }

    // CPU_L1 takes the data from a Float32 texture and computes L1 on CPU
    public func CPU_L1(texture : MTLTexture) -> Float {
        var sum : Float = 0;
        for c in 0...texture.arrayLength-1 {
            var buf = Array<Float>(count: texture.width * texture.height, repeatedValue: 0)
            texture.getBytes(&buf, bytesPerRow: texture.width * 4, bytesPerImage: texture.width*texture.height*4,
                fromRegion: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0, slice: c)
            for i in 0...texture.width*texture.height-1 {
                sum += abs(buf[i])
            }
        }
        return sum
    }

    // L2 computes the sum of squares. It requires a 2d float texture array as an input.
    public func L2(texture : MTLTexture) -> Float {
        let commandBuffer = commandQueue!.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(computeL2State!)
        commandEncoder.setTexture(texture, atIndex: 0)
        let resBuf = metalDevice!.newBufferWithLength(4 * texture.height, options: .StorageModeShared)
        commandEncoder.setBuffer(resBuf, offset: 0, atIndex: 0)
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let threadgroupsPerGrid = MTLSizeMake(
            (texture.height + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()
        let resArr = UnsafeMutablePointer<Float>(resBuf.contents())
        var res : Float = 0;
        for i in 0...texture.height - 1 {
            res += resArr[i]
        }
        return res
    }

    // Max computes the max value in the texture. It requires a 2d float texture array as an input.
    func Max(texture : MTLTexture) -> Float {
        let commandBuffer = commandQueue!.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(computeMaxState!)
        commandEncoder.setTexture(texture, atIndex: 0)
        let resBuf = metalDevice!.newBufferWithLength(4 * texture.height, options: .StorageModeShared)
        commandEncoder.setBuffer(resBuf, offset: 0, atIndex: 0)
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let threadgroupsPerGrid = MTLSizeMake(
            (texture.height + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()
        let resArr = UnsafeMutablePointer<Float>(resBuf.contents())
        var res : Float = resArr[0];
        for i in 0...texture.height - 1 {
            if resArr[i] > res {
                res = resArr[i]
            }
        }
        return res
    }

    public func GetResourceAsMetalTexture(named: String) -> MTLTexture? {
        let textureLoader = MTKTextureLoader(device: metalDevice!)
        let input = UIImage(named:named)
        if input == nil {
            return nil
        }
        let cgInput = input!.CGImage
        var txInput: MTLTexture?
        do {
            txInput = try textureLoader.newTextureWithCGImage(cgInput!, options: nil)
        } catch {
            print("GetResourceAsMetalTexture(\(named)): failed to create a metal texture out of CGImage")
        }
        return txInput
    }

    public func Sample8x8(input : MTLTexture) -> [Float] {
        let commandBuffer = commandQueue!.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(sample8x8State!)
        commandEncoder.setTexture(input, atIndex: 0)
        let resBuf = metalDevice!.newBufferWithLength(4 * 8 * 8, options: .StorageModeShared)
        commandEncoder.setBuffer(resBuf, offset: 0, atIndex: 0)
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
        let threadgroupsPerGrid = MTLSizeMake(1, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()

        let resArr = UnsafeMutablePointer<Float>(resBuf.contents())
        var res : [Float] = [Float](count: 8*8, repeatedValue: 0)
        for i in 0...8*8 - 1 {
            res[i] = resArr[i]
        }
        return res
    }

    public func ExtractResult(input: MTLTexture) -> [Float] {
        let commandBuffer = commandQueue!.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        let state = loadKernelState("array1x1_to_buffer_0")
        commandEncoder.setComputePipelineState(state)
        commandEncoder.setTexture(input, atIndex: 0)
        let resBuf = metalDevice!.newBufferWithLength(4 * input.arrayLength, options: .StorageModeShared)
        commandEncoder.setBuffer(resBuf, offset: 0, atIndex: 0)
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let tpgx = (input.arrayLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width;
        let threadgroupsPerGrid = MTLSizeMake(tpgx, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()

        let resArr = UnsafeMutablePointer<Float>(resBuf.contents())
        var res : [Float] = [Float](count: input.arrayLength, repeatedValue: 0)
        for i in 0...input.arrayLength - 1 {
            res[i] = resArr[i]
        }
        return res
    }
}
