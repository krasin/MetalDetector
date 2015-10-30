//
//  Net.swift
//  MetalDetector
//
//  Created by Ivan Krasin on 10/8/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import Foundation
import Metal

func initBlob(device : MTLDevice, channels : Int, height : Int, width : Int) -> MTLTexture {
    let desc = MTLTextureDescriptor()
    desc.textureType = MTLTextureType.Type2DArray
    desc.height = height
    desc.width = width
    desc.pixelFormat = MTLPixelFormat.R32Float
    desc.arrayLength = channels
    return device.newTextureWithDescriptor(desc)
}

func initBufferFromBundle(device : MTLDevice, named: String) -> MTLBuffer {
    let data = getBytesFromBundle(named)
    return device.newBufferWithBytes(data.bytes, length: data.length, options: .StorageModeShared)
}

func subBlob(input : MTLTexture, from : Int, to : Int) -> MTLTexture {
    return input.newTextureViewWithPixelFormat(MTLPixelFormat.R32Float,
            textureType: MTLTextureType.Type2DArray,
            levels: NSMakeRange(0, 1),
            slices: NSMakeRange(from, to-from))
}

public func getBytesFromBundle(named: String) -> NSData {
    let path = NSBundle.mainBundle().pathForResource(named, ofType: "")
    if path == nil {
        print("Resource \(named) not found in the main bundle")
        exit(1)
    }
    let data: NSData? = NSData(contentsOfFile: path!)
    if data == nil {
        print("Could not read from file \(path)")
        exit(1)
    }
    print("\(data!.length) bytes loaded from file \(path)")
    return data!
}

public func loadLabels(named: String) -> [String] {
    let path = NSBundle.mainBundle().pathForResource(named, ofType: "")
    if path == nil {
        print("Resource \(named) not found in the main bundle")
        exit(1)
    }
    do {
        let content = try String(contentsOfFile:path!, encoding: NSUTF8StringEncoding)
        return content.componentsSeparatedByString("\n")
    } catch _ as NSError {
        // TODO: display error message
        print("Failed to load labels from file \(path): <error message to be included>")
        exit(1)
    }
}

public struct NetLayer {
    var name: String
    var weights: String
    var shards: Int
    var top: String
    var bottoms: [String]
}

public protocol NetConfig {
    func GetLayers() -> [NetLayer]
    func CreateBlobs(device: MTLDevice) -> [String: MTLTexture]
    func CreateWeights(device: MTLDevice) -> [String: MTLBuffer]
}

public class Net {
    var engine : Engine
    public var layers : [NetLayer]
    public var blobs : [String: MTLTexture]
    public var weights : [String: MTLBuffer]
    public var labels : [String]

    public var L1 = [String: Float]()
    public var L2 = [String: Float]()

    public init(engine: Engine, config: NetConfig) {
        self.engine = engine
        self.layers = config.GetLayers()
        self.blobs = config.CreateBlobs(engine.metalDevice!)
        self.weights = config.CreateWeights(engine.metalDevice!)
        self.labels = loadLabels("synset_words.txt")
        print("Loaded \(self.labels.count) labels")
    }

    public func forward(input: MTLTexture) -> [Float] {
        let commandBuffer = engine.commandQueue!.commandBuffer()

        engine.Preprocess(commandBuffer, input: input, output: blobs["data"]!)

        for layer in layers {
            if layer.bottoms.count != 1 {
                continue
            }
            var w : MTLBuffer? = nil
            if layer.weights != "" {
                w = weights[layer.weights]
                if w == nil {
                    print("Weights \(layer.weights) for layer \(layer.name) not found")
                    exit(1)
                }
            }
            for i in 0...layer.shards-1 {
                engine.UnaryLayer(commandBuffer,
                    name: "\(layer.name)_\(i)",
                    weights: w,
                    input: blobs[layer.bottoms[0]]!,
                    output: blobs[layer.top]!)
            }
        }

        engine.PerFilterLayer(commandBuffer, name: "loss3_classifier_0",
            weights: weights["loss3_classifier"]!, numFilters: 1000,
            input: blobs["pool5_7x7_s1"]!, output: blobs["loss3_classifier"]!)
        engine.PerFilterLayer(commandBuffer, name: "prob_0",
            weights: nil, numFilters: 1,
            input: blobs["loss3_classifier"]!, output: blobs["prob"]!)

        commandBuffer.commit();
        commandBuffer.waitUntilCompleted()

        //print("Status: \(commandBuffer.status.rawValue)")
        //print("May be error: \(commandBuffer.error)")

        //printStats("data", blob: data)
        //printStats("conv1_7x7_s2", blob: conv1_7x7_s2)
        var res = Array<Float>(count: 1000, repeatedValue: 0)
        for i in 0...res.count-1 {
            blobs["prob"]!.getBytes(&res[i], bytesPerRow: 4, bytesPerImage: 4,
                fromRegion: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: i)
        }
        return res
    }
}
