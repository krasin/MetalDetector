//
//  MetalDetectorTests.swift
//  MetalDetectorTests
//
//  Created by Ivan Krasin on 10/1/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import XCTest
import Metal
import MetalKit
import MetalPerformanceShaders
import UIKit
import MetalDetector


class MetalDetectorTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testInitMetal() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        print("Hello, I am a test!")
        let metalDevice = MTLCreateSystemDefaultDevice()
        XCTAssert(metalDevice != nil)
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

    func testGoogleNetOnCat() {
        let engine = Engine()
        let net = Net(engine: engine, config: GoogLeNetConfig())
        let input = engine.GetResourceAsMetalTexture("cat.png")
        XCTAssert(input != nil)
        if input == nil {
            return
        }
        //print("input: h: \(input!.height) w: \(input!.width) textureType: \(input!.textureType.rawValue) pixelFormat: \(input!.pixelFormat.rawValue)")
        XCTAssert(input != nil)

        var ans = net.forward(input!)
        // HACK: find the answer
        for i in 1...5 {
            let (idx, p) = argMax(ans)
            ans[idx] = 0
            print("\(i). \(net.labels[idx]) - \(p)")
        }
        // data: L1: 8815415.000000, L2: 646363264.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["data"]!), Float(8815415), accuracy: Float(8815415 / 10000))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["data"]!), Float(646363264), accuracy: Float(646363264 / 1000))

        let cpu_l1 = engine.CPU_L1(net.blobs["data"]!)
        XCTAssertEqualWithAccuracy(cpu_l1, Float(8815415), accuracy:Float(8815415 / 1000))

        // conv1/7x7_s2: L1: 26894028, L2: 7389178368
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["conv1_7x7_s2"]!), Float(26894028), accuracy: Float(26894028 / 200))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["conv1_7x7_s2"]!), Float(7389178368), accuracy: Float(7389178368 / 100))

        // pool1/3x3_s2 L1: 15706585.000000, L2: 5238782464.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["pool1_3x3_s2"]!), Float(15706585), accuracy: Float(15706585 / 200))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["pool1_3x3_s2"]!), Float(5238782464), accuracy: Float(5238782464 / 100))

        // pool1/norm1: L1: 6763517.000000, L2: 470019456.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["pool1_norm1"]!), Float(6763517), accuracy: Float(6763517 / 200))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["pool1_norm1"]!), Float(470019456), accuracy: Float(470019456 / 100))

        // conv2/3x3_reduce: 1 64 56 56 (200704), L1: 4607045.000000, L2: 365177216.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["conv2_3x3_reduce"]!), Float(4607045), accuracy: Float(4607045 / 200))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["conv2_3x3_reduce"]!), Float(365177216), accuracy: Float(365177216 / 100))

        // conv2/3x3: 1 192 56 56 (602112), L1: 6190394.000000, L2: 651581568.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["conv2_3x3"]!), Float(6190394), accuracy: Float(6190394 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["conv2_3x3"]!), Float(651581568), accuracy: Float(651581568 / 100))

        // conv2/norm2: 1 192 56 56 (602112), L1: 4979390.500000, L2: 350693696.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["conv2_norm2"]!), Float(4979390), accuracy: Float(4979390 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["conv2_norm2"]!), Float(350693696), accuracy: Float(350693696 / 100))

        // pool2/3x3_s2: 1 192 28 28 (150528), L1: 3983016.000000, L2: 300735104.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["pool2_3x3_s2"]!), Float(3983016), accuracy: Float(3983016 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["pool2_3x3_s2"]!), Float(300735104), accuracy: Float(300735104 / 100))

        // inception_3a/1x1: 1 64 28 28 (50176), L1: 1376699.000000, L2: 133712880.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_1x1"]!), Float(1376699), accuracy: Float(1376699 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_1x1"]!), Float(133712880), accuracy: Float(133712880 / 100))

        // inception_3a/3x3_reduce: 1 96 28 28 (75264), L1: 1716600.250000, L2: 141276912.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_3x3_reduce"]!), Float(1716600), accuracy: Float(1716600 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_3x3_reduce"]!), Float(141276912), accuracy: Float(141276912 / 50))

        // inception_3a/3x3: 1 128 28 28 (100352), L1: 2320588.250000, L2: 287686272.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_3x3"]!), Float(2320588), accuracy: Float(2320588 / 50))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_3x3"]!), Float(287686272), accuracy: Float(287686272 / 25))

        // inception_3a/5x5_reduce: 1 16 28 28 (12544), L1: 418858.343750, L2: 37575728.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_5x5_reduce"]!), Float(418858), accuracy: Float(418858 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_5x5_reduce"]!), Float(37575728), accuracy: Float(37575728 / 50))

        // inception_3a/5x5: 1 32 28 28 (25088), L1: 697953.750000, L2: 80932752.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_5x5"]!), Float(697953), accuracy: Float(697953 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_5x5"]!), Float(80932752), accuracy: Float(80932752 / 100))

        // inception_3a/pool: 1 192 28 28 (150528), L1: 8837156.000000, L2: 766142080.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_pool"]!), Float(8837156), accuracy: Float(8837156 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_pool"]!), Float(766142080), accuracy: Float(766142080 / 100))

        // inception_3a/pool_proj: 1 32 28 28 (25088), L1: 460720.125000, L2: 39634208.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_pool_proj"]!), Float(460720), accuracy: Float(460720 / 50))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_pool_proj"]!), Float(39634208), accuracy: Float(39634208 / 25))

        // inception_3a/output: 1 256 28 28 (200704), L1: 4855965.000000, L2: 541964032.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3a_output"]!), Float(4855965), accuracy: Float(4855965 / 100))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3a_output"]!), Float(541964032), accuracy: Float(541964032 / 50))

        // inception_3b/output: 1 480 28 28 (376320), L1: 2070707.625000, L2: 173486784.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_3b_output"]!), Float(2070707), accuracy: Float(2070707 / 50))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_3b_output"]!), Float(173486784), accuracy: Float(173486784 / 25))

        // inception_4a/output: 1 512 14 14 (100352), L1: 803036.125000, L2: 76048768.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_4a_output"]!), Float(803036), accuracy: Float(803036 / 25))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_4a_output"]!), Float(76048768), accuracy: Float(76048768 / 25))

        // inception_4b/output: 1 512 14 14 (100352), L1: 1172061.000000, L2: 77093112.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_4b_output"]!), Float(1172061), accuracy: Float(1172061 / 25))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_4b_output"]!), Float(77093112), accuracy: Float(77093112 / 25))

        // inception_4c/output: 1 512 14 14 (100352), L1: 907352.500000, L2: 62197672.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_4c_output"]!), Float(907352), accuracy: Float(907352 / 25))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_4c_output"]!), Float(62197672), accuracy: Float(62197672 / 25))

        // inception_4d/output: 1 528 14 14 (103488), L1: 464165.843750, L2: 27689768.000000
        XCTAssertEqualWithAccuracy(engine.L1(net.blobs["inception_4d_output"]!), Float(464165), accuracy: Float(464165 / 20))
        XCTAssertEqualWithAccuracy(engine.L2(net.blobs["inception_4d_output"]!), Float(27689768), accuracy: Float(27689768 / 10))

        /*print("Pool1_3x3_s2: ")
        var buf = Array<Float>(count: 56 * 56, repeatedValue: 0)
        net.blobs["pool1_3x3_s2"]!.getBytes(&buf, bytesPerRow: 56 * 4, bytesPerImage: 56 * 56 * 4,
            fromRegion: MTLRegionMake2D(0, 0, 56, 56), mipmapLevel: 0, slice: 0)
        for j in 0...6 {
            for i in 0...16 {
                print("\(buf[j*56+i])  ", terminator:"")
            }
            print("")
        }
        print("")
        print("")*/

        /*print("data: h: \(net.data.height) w: \(net.data.width) textureType: \(net.data.textureType.rawValue) pixelFormat: \(net.data.pixelFormat.rawValue)")
        for c in 0...2 {
            print("Data, c=\(c)")
            var buf0 = Array<Float>(count: 224 * 224 * 4, repeatedValue: 0)
            net.data.getBytes(&buf0, bytesPerRow: 224 * 4, bytesPerImage: 224*224*4,
                fromRegion: MTLRegionMake2D(0, 0, 224, 224), mipmapLevel: 0, slice: c)
            for j in 0...6 {
                for i in 0...6 {
                    print("\(buf0[j*224+i])  ", terminator:"")
                }
                print("")
            }
            print("")
            print("")
        }

        print("Sample8x8 for data: ")
        var sample = engine.Sample8x8(net.data)
        for j in 0...7 {
            for i in 0...7 {
                print("\(sample[j*8+i])  ", terminator:"")
            }
            print("")
        }
        print("")
        print("")*/

        /*var buf = Array<Float>(count: 112 * 112 * 4, repeatedValue: 0)
        net.conv1_7x7_s2.getBytes(&buf, bytesPerRow: 112*4, bytesPerImage: 112*112*4,
            fromRegion: MTLRegionMake2D(0, 0, 112, 112), mipmapLevel: 0, slice: 31)

        print("buf[341]=\(buf[341])")
        for j in 100...106 {
            for i in 100...106 {
                print("\(buf[j*112+i])  ", terminator:"")
            }
            print("")
        }*/

        self.measureBlock {
            for _ in 1...1 {
                net.forward(input!)
            }
        }
    }

}
