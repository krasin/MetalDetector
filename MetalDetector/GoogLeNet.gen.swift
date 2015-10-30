 // This file is generated. Do not edit unless you have no access to the generator.

import Foundation
import Metal

public class GoogLeNetConfig : NetConfig {
    public init() {}

    public func CreateBlobs(device: MTLDevice) -> [String: MTLTexture] {
        var blobs : [String: MTLTexture] = [:]
        blobs["data"] = initBlob(device, channels: 3, height: 224, width: 224)
        blobs["conv1_7x7_s2"] = initBlob(device, channels: 64, height: 112, width: 112)
        blobs["pool1_3x3_s2"] = initBlob(device, channels: 64, height: 56, width: 56)
        blobs["pool1_norm1"] = initBlob(device, channels: 64, height: 56, width: 56)
        blobs["conv2_3x3_reduce"] = initBlob(device, channels: 64, height: 56, width: 56)
        blobs["conv2_3x3"] = initBlob(device, channels: 192, height: 56, width: 56)
        blobs["conv2_norm2"] = initBlob(device, channels: 192, height: 56, width: 56)
        blobs["pool2_3x3_s2"] = initBlob(device, channels: 192, height: 28, width: 28)
        blobs["inception_3a_output"] = initBlob(device, channels: 256, height: 28, width: 28)
        blobs["inception_3a_1x1"] = subBlob(blobs["inception_3a_output"]!, from: 0, to: 64) // 64
        blobs["inception_3a_3x3_reduce"] = initBlob(device, channels: 96, height: 28, width: 28)
        blobs["inception_3a_3x3"] = subBlob(blobs["inception_3a_output"]!, from: 64, to: 192) // 128
        blobs["inception_3a_5x5_reduce"] = initBlob(device, channels: 16, height: 28, width: 28)
        blobs["inception_3a_5x5"] = subBlob(blobs["inception_3a_output"]!, from: 192, to: 224) // 32
        blobs["inception_3a_pool"] = initBlob(device, channels: 192, height: 28, width: 28)
        blobs["inception_3a_pool_proj"] = subBlob(blobs["inception_3a_output"]!, from: 224, to: 256) // 32
        blobs["inception_3b_output"] = initBlob(device, channels: 480, height: 28, width: 28)
        blobs["inception_3b_1x1"] = subBlob(blobs["inception_3b_output"]!, from: 0, to: 128) // 128
        blobs["inception_3b_3x3_reduce"] = initBlob(device, channels: 128, height: 28, width: 28)
        blobs["inception_3b_3x3"] = subBlob(blobs["inception_3b_output"]!, from: 128, to: 320) // 192
        blobs["inception_3b_5x5_reduce"] = initBlob(device, channels: 32, height: 28, width: 28)
        blobs["inception_3b_5x5"] = subBlob(blobs["inception_3b_output"]!, from: 320, to: 416) // 96
        blobs["inception_3b_pool"] = initBlob(device, channels: 256, height: 28, width: 28)
        blobs["inception_3b_pool_proj"] = subBlob(blobs["inception_3b_output"]!, from: 416, to: 480) // 64
        blobs["pool3_3x3_s2"] = initBlob(device, channels: 480, height: 14, width: 14)
        blobs["inception_4a_output"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4a_1x1"] = subBlob(blobs["inception_4a_output"]!, from: 0, to: 192) // 192
        blobs["inception_4a_3x3_reduce"] = initBlob(device, channels: 96, height: 14, width: 14)
        blobs["inception_4a_3x3"] = subBlob(blobs["inception_4a_output"]!, from: 192, to: 400) // 208
        blobs["inception_4a_5x5_reduce"] = initBlob(device, channels: 16, height: 14, width: 14)
        blobs["inception_4a_5x5"] = subBlob(blobs["inception_4a_output"]!, from: 400, to: 448) // 48
        blobs["inception_4a_pool"] = initBlob(device, channels: 480, height: 14, width: 14)
        blobs["inception_4a_pool_proj"] = subBlob(blobs["inception_4a_output"]!, from: 448, to: 512) // 64
        blobs["inception_4b_output"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4b_1x1"] = subBlob(blobs["inception_4b_output"]!, from: 0, to: 160) // 160
        blobs["inception_4b_3x3_reduce"] = initBlob(device, channels: 112, height: 14, width: 14)
        blobs["inception_4b_3x3"] = subBlob(blobs["inception_4b_output"]!, from: 160, to: 384) // 224
        blobs["inception_4b_5x5_reduce"] = initBlob(device, channels: 24, height: 14, width: 14)
        blobs["inception_4b_5x5"] = subBlob(blobs["inception_4b_output"]!, from: 384, to: 448) // 64
        blobs["inception_4b_pool"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4b_pool_proj"] = subBlob(blobs["inception_4b_output"]!, from: 448, to: 512) // 64
        blobs["inception_4c_output"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4c_1x1"] = subBlob(blobs["inception_4c_output"]!, from: 0, to: 128) // 128
        blobs["inception_4c_3x3_reduce"] = initBlob(device, channels: 128, height: 14, width: 14)
        blobs["inception_4c_3x3"] = subBlob(blobs["inception_4c_output"]!, from: 128, to: 384) // 256
        blobs["inception_4c_5x5_reduce"] = initBlob(device, channels: 24, height: 14, width: 14)
        blobs["inception_4c_5x5"] = subBlob(blobs["inception_4c_output"]!, from: 384, to: 448) // 64
        blobs["inception_4c_pool"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4c_pool_proj"] = subBlob(blobs["inception_4c_output"]!, from: 448, to: 512) // 64
        blobs["inception_4d_output"] = initBlob(device, channels: 528, height: 14, width: 14)
        blobs["inception_4d_1x1"] = subBlob(blobs["inception_4d_output"]!, from: 0, to: 112) // 112
        blobs["inception_4d_3x3_reduce"] = initBlob(device, channels: 144, height: 14, width: 14)
        blobs["inception_4d_3x3"] = subBlob(blobs["inception_4d_output"]!, from: 112, to: 400) // 288
        blobs["inception_4d_5x5_reduce"] = initBlob(device, channels: 32, height: 14, width: 14)
        blobs["inception_4d_5x5"] = subBlob(blobs["inception_4d_output"]!, from: 400, to: 464) // 64
        blobs["inception_4d_pool"] = initBlob(device, channels: 512, height: 14, width: 14)
        blobs["inception_4d_pool_proj"] = subBlob(blobs["inception_4d_output"]!, from: 464, to: 528) // 64
        blobs["inception_4e_output"] = initBlob(device, channels: 832, height: 14, width: 14)
        blobs["inception_4e_1x1"] = subBlob(blobs["inception_4e_output"]!, from: 0, to: 256) // 256
        blobs["inception_4e_3x3_reduce"] = initBlob(device, channels: 160, height: 14, width: 14)
        blobs["inception_4e_3x3"] = subBlob(blobs["inception_4e_output"]!, from: 256, to: 576) // 320
        blobs["inception_4e_5x5_reduce"] = initBlob(device, channels: 32, height: 14, width: 14)
        blobs["inception_4e_5x5"] = subBlob(blobs["inception_4e_output"]!, from: 576, to: 704) // 128
        blobs["inception_4e_pool"] = initBlob(device, channels: 528, height: 14, width: 14)
        blobs["inception_4e_pool_proj"] = subBlob(blobs["inception_4e_output"]!, from: 704, to: 832) // 128
        blobs["pool4_3x3_s2"] = initBlob(device, channels: 832, height: 7, width: 7)
        blobs["inception_5a_output"] = initBlob(device, channels: 832, height: 7, width: 7)
        blobs["inception_5a_1x1"] = subBlob(blobs["inception_5a_output"]!, from: 0, to: 256) // 256
        blobs["inception_5a_3x3_reduce"] = initBlob(device, channels: 160, height: 7, width: 7)
        blobs["inception_5a_3x3"] = subBlob(blobs["inception_5a_output"]!, from: 256, to: 576) // 320
        blobs["inception_5a_5x5_reduce"] = initBlob(device, channels: 32, height: 7, width: 7)
        blobs["inception_5a_5x5"] = subBlob(blobs["inception_5a_output"]!, from: 576, to: 704) // 128
        blobs["inception_5a_pool"] = initBlob(device, channels: 832, height: 7, width: 7)
        blobs["inception_5a_pool_proj"] = subBlob(blobs["inception_5a_output"]!, from: 704, to: 832) // 128
        blobs["inception_5b_output"] = initBlob(device, channels: 1024, height: 7, width: 7)
        blobs["inception_5b_1x1"] = subBlob(blobs["inception_5b_output"]!, from: 0, to: 384) // 384
        blobs["inception_5b_3x3_reduce"] = initBlob(device, channels: 192, height: 7, width: 7)
        blobs["inception_5b_3x3"] = subBlob(blobs["inception_5b_output"]!, from: 384, to: 768) // 384
        blobs["inception_5b_5x5_reduce"] = initBlob(device, channels: 48, height: 7, width: 7)
        blobs["inception_5b_5x5"] = subBlob(blobs["inception_5b_output"]!, from: 768, to: 896) // 128
        blobs["inception_5b_pool"] = initBlob(device, channels: 832, height: 7, width: 7)
        blobs["inception_5b_pool_proj"] = subBlob(blobs["inception_5b_output"]!, from: 896, to: 1024) // 128
        blobs["pool5_7x7_s1"] = initBlob(device, channels: 1024, height: 1, width: 1)
        blobs["loss3_classifier"] = initBlob(device, channels: 1000, height: 1, width: 1)
        blobs["prob"] = initBlob(device, channels: 1000, height: 1, width: 1)
        return blobs
    }

    public func CreateWeights(device: MTLDevice) -> [String: MTLBuffer] {
        var res: [String: MTLBuffer] = [:]
        let data = getBytesFromBundle("GoogLeNet.data")
        let ptr = UnsafePointer<Float>(data.bytes)
        res["conv1_7x7_s2"] = device.newBufferWithBytes(ptr.advancedBy(0), length: 37632, options: .StorageModeShared)
        res["conv2_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(9408), length: 16384, options: .StorageModeShared)
        res["conv2_3x3"] = device.newBufferWithBytes(ptr.advancedBy(13504), length: 442368, options: .StorageModeShared)
        res["inception_3a_1x1"] = device.newBufferWithBytes(ptr.advancedBy(124096), length: 49152, options: .StorageModeShared)
        res["inception_3a_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(136384), length: 73728, options: .StorageModeShared)
        res["inception_3a_3x3"] = device.newBufferWithBytes(ptr.advancedBy(154816), length: 442368, options: .StorageModeShared)
        res["inception_3a_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(265408), length: 12288, options: .StorageModeShared)
        res["inception_3a_5x5"] = device.newBufferWithBytes(ptr.advancedBy(268480), length: 51200, options: .StorageModeShared)
        res["inception_3a_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(281280), length: 24576, options: .StorageModeShared)
        res["inception_3b_1x1"] = device.newBufferWithBytes(ptr.advancedBy(287424), length: 131072, options: .StorageModeShared)
        res["inception_3b_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(320192), length: 131072, options: .StorageModeShared)
        res["inception_3b_3x3"] = device.newBufferWithBytes(ptr.advancedBy(352960), length: 884736, options: .StorageModeShared)
        res["inception_3b_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(574144), length: 32768, options: .StorageModeShared)
        res["inception_3b_5x5"] = device.newBufferWithBytes(ptr.advancedBy(582336), length: 307200, options: .StorageModeShared)
        res["inception_3b_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(659136), length: 65536, options: .StorageModeShared)
        res["inception_4a_1x1"] = device.newBufferWithBytes(ptr.advancedBy(675520), length: 368640, options: .StorageModeShared)
        res["inception_4a_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(767680), length: 184320, options: .StorageModeShared)
        res["inception_4a_3x3"] = device.newBufferWithBytes(ptr.advancedBy(813760), length: 718848, options: .StorageModeShared)
        res["inception_4a_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(993472), length: 30720, options: .StorageModeShared)
        res["inception_4a_5x5"] = device.newBufferWithBytes(ptr.advancedBy(1001152), length: 76800, options: .StorageModeShared)
        res["inception_4a_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(1020352), length: 122880, options: .StorageModeShared)
        res["inception_4b_1x1"] = device.newBufferWithBytes(ptr.advancedBy(1051072), length: 327680, options: .StorageModeShared)
        res["inception_4b_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(1132992), length: 229376, options: .StorageModeShared)
        res["inception_4b_3x3"] = device.newBufferWithBytes(ptr.advancedBy(1190336), length: 903168, options: .StorageModeShared)
        res["inception_4b_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(1416128), length: 49152, options: .StorageModeShared)
        res["inception_4b_5x5"] = device.newBufferWithBytes(ptr.advancedBy(1428416), length: 153600, options: .StorageModeShared)
        res["inception_4b_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(1466816), length: 131072, options: .StorageModeShared)
        res["inception_4c_1x1"] = device.newBufferWithBytes(ptr.advancedBy(1499584), length: 262144, options: .StorageModeShared)
        res["inception_4c_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(1565120), length: 262144, options: .StorageModeShared)
        res["inception_4c_3x3"] = device.newBufferWithBytes(ptr.advancedBy(1630656), length: 1179648, options: .StorageModeShared)
        res["inception_4c_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(1925568), length: 49152, options: .StorageModeShared)
        res["inception_4c_5x5"] = device.newBufferWithBytes(ptr.advancedBy(1937856), length: 153600, options: .StorageModeShared)
        res["inception_4c_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(1976256), length: 131072, options: .StorageModeShared)
        res["inception_4d_1x1"] = device.newBufferWithBytes(ptr.advancedBy(2009024), length: 229376, options: .StorageModeShared)
        res["inception_4d_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(2066368), length: 294912, options: .StorageModeShared)
        res["inception_4d_3x3"] = device.newBufferWithBytes(ptr.advancedBy(2140096), length: 1492992, options: .StorageModeShared)
        res["inception_4d_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(2513344), length: 65536, options: .StorageModeShared)
        res["inception_4d_5x5"] = device.newBufferWithBytes(ptr.advancedBy(2529728), length: 204800, options: .StorageModeShared)
        res["inception_4d_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(2580928), length: 131072, options: .StorageModeShared)
        res["inception_4e_1x1"] = device.newBufferWithBytes(ptr.advancedBy(2613696), length: 540672, options: .StorageModeShared)
        res["inception_4e_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(2748864), length: 337920, options: .StorageModeShared)
        res["inception_4e_3x3"] = device.newBufferWithBytes(ptr.advancedBy(2833344), length: 1843200, options: .StorageModeShared)
        res["inception_4e_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(3294144), length: 67584, options: .StorageModeShared)
        res["inception_4e_5x5"] = device.newBufferWithBytes(ptr.advancedBy(3311040), length: 409600, options: .StorageModeShared)
        res["inception_4e_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(3413440), length: 270336, options: .StorageModeShared)
        res["inception_5a_1x1"] = device.newBufferWithBytes(ptr.advancedBy(3481024), length: 851968, options: .StorageModeShared)
        res["inception_5a_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(3694016), length: 532480, options: .StorageModeShared)
        res["inception_5a_3x3"] = device.newBufferWithBytes(ptr.advancedBy(3827136), length: 1843200, options: .StorageModeShared)
        res["inception_5a_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(4287936), length: 106496, options: .StorageModeShared)
        res["inception_5a_5x5"] = device.newBufferWithBytes(ptr.advancedBy(4314560), length: 409600, options: .StorageModeShared)
        res["inception_5a_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(4416960), length: 425984, options: .StorageModeShared)
        res["inception_5b_1x1"] = device.newBufferWithBytes(ptr.advancedBy(4523456), length: 1277952, options: .StorageModeShared)
        res["inception_5b_3x3_reduce"] = device.newBufferWithBytes(ptr.advancedBy(4842944), length: 638976, options: .StorageModeShared)
        res["inception_5b_3x3"] = device.newBufferWithBytes(ptr.advancedBy(5002688), length: 2654208, options: .StorageModeShared)
        res["inception_5b_5x5_reduce"] = device.newBufferWithBytes(ptr.advancedBy(5666240), length: 159744, options: .StorageModeShared)
        res["inception_5b_5x5"] = device.newBufferWithBytes(ptr.advancedBy(5706176), length: 614400, options: .StorageModeShared)
        res["inception_5b_pool_proj"] = device.newBufferWithBytes(ptr.advancedBy(5859776), length: 425984, options: .StorageModeShared)
        res["loss3_classifier"] = device.newBufferWithBytes(ptr.advancedBy(5966272), length: 4096000, options: .StorageModeShared)

        return res
    }

    public func GetLayers() -> [NetLayer] {
        return [
            NetLayer(name: "conv1_7x7_s2", weights: "conv1_7x7_s2", shards: 1, top: "conv1_7x7_s2", bottoms: ["data"]),
            NetLayer(name: "pool1_3x3_s2", weights: "", shards: 1, top: "pool1_3x3_s2", bottoms: ["conv1_7x7_s2"]),
            NetLayer(name: "pool1_norm1", weights: "", shards: 1, top: "pool1_norm1", bottoms: ["pool1_3x3_s2"]),
            NetLayer(name: "conv2_3x3_reduce", weights: "conv2_3x3_reduce", shards: 1, top: "conv2_3x3_reduce", bottoms: ["pool1_norm1"]),
            NetLayer(name: "conv2_3x3", weights: "conv2_3x3", shards: 1, top: "conv2_3x3", bottoms: ["conv2_3x3_reduce"]),
            NetLayer(name: "conv2_norm2", weights: "", shards: 1, top: "conv2_norm2", bottoms: ["conv2_3x3"]),
            NetLayer(name: "pool2_3x3_s2", weights: "", shards: 1, top: "pool2_3x3_s2", bottoms: ["conv2_norm2"]),
            NetLayer(name: "inception_3a_1x1", weights: "inception_3a_1x1", shards: 1, top: "inception_3a_1x1", bottoms: ["pool2_3x3_s2"]),
            NetLayer(name: "inception_3a_3x3_reduce", weights: "inception_3a_3x3_reduce", shards: 1, top: "inception_3a_3x3_reduce", bottoms: ["pool2_3x3_s2"]),
            NetLayer(name: "inception_3a_3x3", weights: "inception_3a_3x3", shards: 1, top: "inception_3a_3x3", bottoms: ["inception_3a_3x3_reduce"]),
            NetLayer(name: "inception_3a_5x5_reduce", weights: "inception_3a_5x5_reduce", shards: 1, top: "inception_3a_5x5_reduce", bottoms: ["pool2_3x3_s2"]),
            NetLayer(name: "inception_3a_5x5", weights: "inception_3a_5x5", shards: 1, top: "inception_3a_5x5", bottoms: ["inception_3a_5x5_reduce"]),
            NetLayer(name: "inception_3a_pool", weights: "", shards: 1, top: "inception_3a_pool", bottoms: ["pool2_3x3_s2"]),
            NetLayer(name: "inception_3a_pool_proj", weights: "inception_3a_pool_proj", shards: 1, top: "inception_3a_pool_proj", bottoms: ["inception_3a_pool"]),
            NetLayer(name: "inception_3a_output", weights: "", shards: 1, top: "inception_3a_output", bottoms: ["inception_3a_1x1", "inception_3a_3x3", "inception_3a_5x5", "inception_3a_pool_proj"]),
            NetLayer(name: "inception_3b_1x1", weights: "inception_3b_1x1", shards: 1, top: "inception_3b_1x1", bottoms: ["inception_3a_output"]),
            NetLayer(name: "inception_3b_3x3_reduce", weights: "inception_3b_3x3_reduce", shards: 1, top: "inception_3b_3x3_reduce", bottoms: ["inception_3a_output"]),
            NetLayer(name: "inception_3b_3x3", weights: "inception_3b_3x3", shards: 1, top: "inception_3b_3x3", bottoms: ["inception_3b_3x3_reduce"]),
            NetLayer(name: "inception_3b_5x5_reduce", weights: "inception_3b_5x5_reduce", shards: 1, top: "inception_3b_5x5_reduce", bottoms: ["inception_3a_output"]),
            NetLayer(name: "inception_3b_5x5", weights: "inception_3b_5x5", shards: 1, top: "inception_3b_5x5", bottoms: ["inception_3b_5x5_reduce"]),
            NetLayer(name: "inception_3b_pool", weights: "", shards: 1, top: "inception_3b_pool", bottoms: ["inception_3a_output"]),
            NetLayer(name: "inception_3b_pool_proj", weights: "inception_3b_pool_proj", shards: 1, top: "inception_3b_pool_proj", bottoms: ["inception_3b_pool"]),
            NetLayer(name: "inception_3b_output", weights: "", shards: 1, top: "inception_3b_output", bottoms: ["inception_3b_1x1", "inception_3b_3x3", "inception_3b_5x5", "inception_3b_pool_proj"]),
            NetLayer(name: "pool3_3x3_s2", weights: "", shards: 1, top: "pool3_3x3_s2", bottoms: ["inception_3b_output"]),
            NetLayer(name: "inception_4a_1x1", weights: "inception_4a_1x1", shards: 1, top: "inception_4a_1x1", bottoms: ["pool3_3x3_s2"]),
            NetLayer(name: "inception_4a_3x3_reduce", weights: "inception_4a_3x3_reduce", shards: 1, top: "inception_4a_3x3_reduce", bottoms: ["pool3_3x3_s2"]),
            NetLayer(name: "inception_4a_3x3", weights: "inception_4a_3x3", shards: 1, top: "inception_4a_3x3", bottoms: ["inception_4a_3x3_reduce"]),
            NetLayer(name: "inception_4a_5x5_reduce", weights: "inception_4a_5x5_reduce", shards: 1, top: "inception_4a_5x5_reduce", bottoms: ["pool3_3x3_s2"]),
            NetLayer(name: "inception_4a_5x5", weights: "inception_4a_5x5", shards: 1, top: "inception_4a_5x5", bottoms: ["inception_4a_5x5_reduce"]),
            NetLayer(name: "inception_4a_pool", weights: "", shards: 1, top: "inception_4a_pool", bottoms: ["pool3_3x3_s2"]),
            NetLayer(name: "inception_4a_pool_proj", weights: "inception_4a_pool_proj", shards: 1, top: "inception_4a_pool_proj", bottoms: ["inception_4a_pool"]),
            NetLayer(name: "inception_4a_output", weights: "", shards: 1, top: "inception_4a_output", bottoms: ["inception_4a_1x1", "inception_4a_3x3", "inception_4a_5x5", "inception_4a_pool_proj"]),
            NetLayer(name: "inception_4b_1x1", weights: "inception_4b_1x1", shards: 1, top: "inception_4b_1x1", bottoms: ["inception_4a_output"]),
            NetLayer(name: "inception_4b_3x3_reduce", weights: "inception_4b_3x3_reduce", shards: 1, top: "inception_4b_3x3_reduce", bottoms: ["inception_4a_output"]),
            NetLayer(name: "inception_4b_3x3", weights: "inception_4b_3x3", shards: 1, top: "inception_4b_3x3", bottoms: ["inception_4b_3x3_reduce"]),
            NetLayer(name: "inception_4b_5x5_reduce", weights: "inception_4b_5x5_reduce", shards: 1, top: "inception_4b_5x5_reduce", bottoms: ["inception_4a_output"]),
            NetLayer(name: "inception_4b_5x5", weights: "inception_4b_5x5", shards: 1, top: "inception_4b_5x5", bottoms: ["inception_4b_5x5_reduce"]),
            NetLayer(name: "inception_4b_pool", weights: "", shards: 1, top: "inception_4b_pool", bottoms: ["inception_4a_output"]),
            NetLayer(name: "inception_4b_pool_proj", weights: "inception_4b_pool_proj", shards: 1, top: "inception_4b_pool_proj", bottoms: ["inception_4b_pool"]),
            NetLayer(name: "inception_4b_output", weights: "", shards: 1, top: "inception_4b_output", bottoms: ["inception_4b_1x1", "inception_4b_3x3", "inception_4b_5x5", "inception_4b_pool_proj"]),
            NetLayer(name: "inception_4c_1x1", weights: "inception_4c_1x1", shards: 1, top: "inception_4c_1x1", bottoms: ["inception_4b_output"]),
            NetLayer(name: "inception_4c_3x3_reduce", weights: "inception_4c_3x3_reduce", shards: 1, top: "inception_4c_3x3_reduce", bottoms: ["inception_4b_output"]),
            NetLayer(name: "inception_4c_3x3", weights: "inception_4c_3x3", shards: 1, top: "inception_4c_3x3", bottoms: ["inception_4c_3x3_reduce"]),
            NetLayer(name: "inception_4c_5x5_reduce", weights: "inception_4c_5x5_reduce", shards: 1, top: "inception_4c_5x5_reduce", bottoms: ["inception_4b_output"]),
            NetLayer(name: "inception_4c_5x5", weights: "inception_4c_5x5", shards: 1, top: "inception_4c_5x5", bottoms: ["inception_4c_5x5_reduce"]),
            NetLayer(name: "inception_4c_pool", weights: "", shards: 1, top: "inception_4c_pool", bottoms: ["inception_4b_output"]),
            NetLayer(name: "inception_4c_pool_proj", weights: "inception_4c_pool_proj", shards: 1, top: "inception_4c_pool_proj", bottoms: ["inception_4c_pool"]),
            NetLayer(name: "inception_4c_output", weights: "", shards: 1, top: "inception_4c_output", bottoms: ["inception_4c_1x1", "inception_4c_3x3", "inception_4c_5x5", "inception_4c_pool_proj"]),
            NetLayer(name: "inception_4d_1x1", weights: "inception_4d_1x1", shards: 1, top: "inception_4d_1x1", bottoms: ["inception_4c_output"]),
            NetLayer(name: "inception_4d_3x3_reduce", weights: "inception_4d_3x3_reduce", shards: 1, top: "inception_4d_3x3_reduce", bottoms: ["inception_4c_output"]),
            NetLayer(name: "inception_4d_3x3", weights: "inception_4d_3x3", shards: 1, top: "inception_4d_3x3", bottoms: ["inception_4d_3x3_reduce"]),
            NetLayer(name: "inception_4d_5x5_reduce", weights: "inception_4d_5x5_reduce", shards: 1, top: "inception_4d_5x5_reduce", bottoms: ["inception_4c_output"]),
            NetLayer(name: "inception_4d_5x5", weights: "inception_4d_5x5", shards: 1, top: "inception_4d_5x5", bottoms: ["inception_4d_5x5_reduce"]),
            NetLayer(name: "inception_4d_pool", weights: "", shards: 1, top: "inception_4d_pool", bottoms: ["inception_4c_output"]),
            NetLayer(name: "inception_4d_pool_proj", weights: "inception_4d_pool_proj", shards: 1, top: "inception_4d_pool_proj", bottoms: ["inception_4d_pool"]),
            NetLayer(name: "inception_4d_output", weights: "", shards: 1, top: "inception_4d_output", bottoms: ["inception_4d_1x1", "inception_4d_3x3", "inception_4d_5x5", "inception_4d_pool_proj"]),
            NetLayer(name: "inception_4e_1x1", weights: "inception_4e_1x1", shards: 1, top: "inception_4e_1x1", bottoms: ["inception_4d_output"]),
            NetLayer(name: "inception_4e_3x3_reduce", weights: "inception_4e_3x3_reduce", shards: 1, top: "inception_4e_3x3_reduce", bottoms: ["inception_4d_output"]),
            NetLayer(name: "inception_4e_3x3", weights: "inception_4e_3x3", shards: 1, top: "inception_4e_3x3", bottoms: ["inception_4e_3x3_reduce"]),
            NetLayer(name: "inception_4e_5x5_reduce", weights: "inception_4e_5x5_reduce", shards: 1, top: "inception_4e_5x5_reduce", bottoms: ["inception_4d_output"]),
            NetLayer(name: "inception_4e_5x5", weights: "inception_4e_5x5", shards: 1, top: "inception_4e_5x5", bottoms: ["inception_4e_5x5_reduce"]),
            NetLayer(name: "inception_4e_pool", weights: "", shards: 1, top: "inception_4e_pool", bottoms: ["inception_4d_output"]),
            NetLayer(name: "inception_4e_pool_proj", weights: "inception_4e_pool_proj", shards: 1, top: "inception_4e_pool_proj", bottoms: ["inception_4e_pool"]),
            NetLayer(name: "inception_4e_output", weights: "", shards: 1, top: "inception_4e_output", bottoms: ["inception_4e_1x1", "inception_4e_3x3", "inception_4e_5x5", "inception_4e_pool_proj"]),
            NetLayer(name: "pool4_3x3_s2", weights: "", shards: 1, top: "pool4_3x3_s2", bottoms: ["inception_4e_output"]),
            NetLayer(name: "inception_5a_1x1", weights: "inception_5a_1x1", shards: 1, top: "inception_5a_1x1", bottoms: ["pool4_3x3_s2"]),
            NetLayer(name: "inception_5a_3x3_reduce", weights: "inception_5a_3x3_reduce", shards: 1, top: "inception_5a_3x3_reduce", bottoms: ["pool4_3x3_s2"]),
            NetLayer(name: "inception_5a_3x3", weights: "inception_5a_3x3", shards: 1, top: "inception_5a_3x3", bottoms: ["inception_5a_3x3_reduce"]),
            NetLayer(name: "inception_5a_5x5_reduce", weights: "inception_5a_5x5_reduce", shards: 1, top: "inception_5a_5x5_reduce", bottoms: ["pool4_3x3_s2"]),
            NetLayer(name: "inception_5a_5x5", weights: "inception_5a_5x5", shards: 1, top: "inception_5a_5x5", bottoms: ["inception_5a_5x5_reduce"]),
            NetLayer(name: "inception_5a_pool", weights: "", shards: 1, top: "inception_5a_pool", bottoms: ["pool4_3x3_s2"]),
            NetLayer(name: "inception_5a_pool_proj", weights: "inception_5a_pool_proj", shards: 1, top: "inception_5a_pool_proj", bottoms: ["inception_5a_pool"]),
            NetLayer(name: "inception_5a_output", weights: "", shards: 1, top: "inception_5a_output", bottoms: ["inception_5a_1x1", "inception_5a_3x3", "inception_5a_5x5", "inception_5a_pool_proj"]),
            NetLayer(name: "inception_5b_1x1", weights: "inception_5b_1x1", shards: 1, top: "inception_5b_1x1", bottoms: ["inception_5a_output"]),
            NetLayer(name: "inception_5b_3x3_reduce", weights: "inception_5b_3x3_reduce", shards: 1, top: "inception_5b_3x3_reduce", bottoms: ["inception_5a_output"]),
            NetLayer(name: "inception_5b_3x3", weights: "inception_5b_3x3", shards: 1, top: "inception_5b_3x3", bottoms: ["inception_5b_3x3_reduce"]),
            NetLayer(name: "inception_5b_5x5_reduce", weights: "inception_5b_5x5_reduce", shards: 1, top: "inception_5b_5x5_reduce", bottoms: ["inception_5a_output"]),
            NetLayer(name: "inception_5b_5x5", weights: "inception_5b_5x5", shards: 1, top: "inception_5b_5x5", bottoms: ["inception_5b_5x5_reduce"]),
            NetLayer(name: "inception_5b_pool", weights: "", shards: 1, top: "inception_5b_pool", bottoms: ["inception_5a_output"]),
            NetLayer(name: "inception_5b_pool_proj", weights: "inception_5b_pool_proj", shards: 1, top: "inception_5b_pool_proj", bottoms: ["inception_5b_pool"]),
            NetLayer(name: "inception_5b_output", weights: "", shards: 1, top: "inception_5b_output", bottoms: ["inception_5b_1x1", "inception_5b_3x3", "inception_5b_5x5", "inception_5b_pool_proj"]),
            NetLayer(name: "pool5_7x7_s1", weights: "", shards: 1, top: "pool5_7x7_s1", bottoms: ["inception_5b_output"]),
            // NetLayer(name: "loss3_classifier", weights: "loss3_classifier", shards: 1, top: "loss3_classifier", bottoms: ["pool5_7x7_s1"]),
            // NetLayer(name: "prob", weights: "", shards: 1, top: "prob", bottoms: ["loss3_classifier"]),
        ]
    }
}
