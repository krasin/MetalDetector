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
    public var name: String
    public var weights: String
    public var shards: Int
    public var top: String
    public var bottoms: [String]
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
    public var threadsPerThreadgroup: [String: MTLSize]
    public var labels : [String]

    public var L1 = [String: Float]()
    public var L2 = [String: Float]()

    public init(engine: Engine, config: NetConfig, threadsPerThreadgroup: [String: MTLSize]) {
        self.engine = engine
        self.layers = config.GetLayers()
        self.blobs = config.CreateBlobs(engine.metalDevice!)
        self.weights = config.CreateWeights(engine.metalDevice!)
        self.threadsPerThreadgroup = threadsPerThreadgroup
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
                var cell = self.threadsPerThreadgroup[layer.name]
                if cell == nil {
                    cell = MTLSizeMake(16, 16, 1)
                } else {
                    //print("Using profiled cell size for layer \(layer.name): \(cell!.width)x\(cell!.height)x\(cell!.depth)")
                }
                engine.UnaryLayer(commandBuffer,
                    name: "\(layer.name)_\(i)",
                    weights: w,
                    input: blobs[layer.bottoms[0]]!,
                    output: blobs[layer.top]!,
                    threadsPerThreadgroup: cell!)
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

    func FindLayer(layerName: String) -> NetLayer? {
        for layer in layers {
            if layer.name == layerName {
                return layer
            }
        }
        return nil
    }

    func tryLayer(layer: NetLayer, w: MTLBuffer?, threadsPerThreadgroup: MTLSize) -> Double? {
        if threadsPerThreadgroup.depth != 1 {
            return nil
        }
        let total = threadsPerThreadgroup.width * threadsPerThreadgroup.height * threadsPerThreadgroup.depth
        if total > 512 {
            // TODO: dynamically query this parameter.
            return nil
        }
        var info = mach_timebase_info(numer: 0, denom: 0)
        mach_timebase_info(&info)
        let time_base = Double(info.numer) / Double(info.denom)
        var sumWorkTimeNs: Double = 0
        for i in -1...10 {
            let commandBuffer = engine.commandQueue!.commandBuffer()
            let startTime = mach_absolute_time()
            for i in 0...layer.shards-1 {
                engine.UnaryLayer(commandBuffer,
                    name: "\(layer.name)_\(i)",
                    weights: w,
                    input: blobs[layer.bottoms[0]]!,
                    output: blobs[layer.top]!,
                    threadsPerThreadgroup: threadsPerThreadgroup)
            }
            commandBuffer.commit();
            commandBuffer.waitUntilCompleted()
            let workTimeNs = Double(mach_absolute_time() - startTime) * time_base
            if commandBuffer.error != nil {
                return nil
            }
            if i > 0 {
                // TODO: exclude outliers, compute std deviation.
                sumWorkTimeNs += workTimeNs
                //let workTimeMsStr = NSString(format: "%.1f", workTimeNs / 1E6)
                //print("tryLayer(\"\(layer.name)\"): \(workTimeMsStr) ms")
            }
        }
        let aveWorkTimeNs = sumWorkTimeNs / 10
        return aveWorkTimeNs
    }

    public func ProfileLayer(layerName: String) {
        let layer = FindLayer(layerName)
        if layer == nil {
            print("ProfileLayer(\"\(layerName)\"): layer not found")
            exit(1)
        }
        var w : MTLBuffer? = nil
        if layer!.weights != "" {
            w = weights[layer!.weights]
            if w == nil {
                print("Weights \(layer!.weights) for layer \(layer!.name) not found")
                exit(1)
            }
        }
        let cells: [MTLSize] = [ MTLSizeMake(16, 16, 1), MTLSizeMake(8, 8, 1), MTLSizeMake(16, 8, 1),
            MTLSizeMake(8, 16, 1), MTLSizeMake(32, 16, 1), MTLSizeMake(16, 32, 1), MTLSizeMake(32, 32, 1),
            MTLSizeMake(4, 4, 1), MTLSizeMake(8, 4, 1), MTLSizeMake(4, 8, 1), MTLSizeMake(3, 3, 1),
            MTLSizeMake(4, 3, 1), MTLSizeMake(3, 4, 1), MTLSizeMake(4, 2, 1), MTLSizeMake(2, 4, 1),
            MTLSizeMake(2, 2, 1), MTLSizeMake(1, 1, 1), MTLSizeMake(2, 1, 1), MTLSizeMake(1, 2, 1),
            MTLSizeMake(3, 1, 1), MTLSizeMake(4, 1, 1)]

        var firstTimeNs: Double = 0
        var minTimeNs: Double = 0
        var bestCell: MTLSize?
        for cell in cells {
            let layerTimeNs = tryLayer(layer!, w: w, threadsPerThreadgroup: cell)
            if layerTimeNs == nil {
                // print("ProfileLayer(\"\(layer!.name)\"), cell: \(cell): failed")
                continue
            }
            if bestCell == nil {
                minTimeNs = layerTimeNs!
                firstTimeNs = layerTimeNs!
                bestCell = cell
                continue
            }
            // We want to be conservative about choosing alternative cell size.
            // There must be at least 5% improvement over the default choice of 16x16x1.
            if 0.95 * firstTimeNs > layerTimeNs && minTimeNs > layerTimeNs {
                minTimeNs = layerTimeNs!
                bestCell = cell
            }
            // let layerTimeMsStr = NSString(format: "%.1f", layerTimeNs! / 1E6)
            // print("ProfileLayer(\"\(layer!.name)\"), cell: \(cell): \(layerTimeMsStr) ms")
        }
        let minTimeMsStr = NSString(format: "%.1f", minTimeNs / 1E6)
        let firstTimeMsStr = NSString(format: "%.1f", firstTimeNs / 1E6)
        //print("\"\(layer!.name)\", best cell: \(bestCell!.width)x\(bestCell!.height)x\(bestCell!.depth), \(minTimeMsStr) ms vs \(firstTimeMsStr) ms for 16x16x1")
        // "inception_5b_pool_proj": MTLSizeMake(3, 3, 1),
        if bestCell!.width != 16 || bestCell!.height != 16 || bestCell!.depth != 1 {
          print("\"\(layer!.name)\": MTLSizeMake(\(bestCell!.width), \(bestCell!.height), \(bestCell!.depth)), // \(minTimeMsStr) ms vs \(firstTimeMsStr) ms for 16x16x1")
        }
    }
}
