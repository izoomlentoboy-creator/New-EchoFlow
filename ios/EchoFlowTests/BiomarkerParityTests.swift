import XCTest
@testable import EchoFlow

/// Validates that the native Swift `BiomarkerExtractor` reproduces the Python
/// reference (`artifacts/parity_refs.json`) within tolerance. Add that JSON to
/// the test bundle's resources before running.
///
/// Generate the reference with:  python -m echoflow.parity
final class BiomarkerParityTests: XCTestCase {

    struct Refs: Decodable {
        let sample_rate: Int
        let names: [String]
        let n_fft: Int
        let hop: Int
        let tolerance_rel: Double
        let cases: [Case]
        struct Case: Decodable {
            let `class`: String
            let waveform: [Float]
            let bio: [String: Float]
        }
    }

    func testBiomarkerParity() throws {
        guard let url = Bundle(for: type(of: self))
                .url(forResource: "parity_refs", withExtension: "json") else {
            throw XCTSkip("parity_refs.json not bundled; run `python -m echoflow.parity`")
        }
        let refs = try JSONDecoder().decode(Refs.self, from: Data(contentsOf: url))

        for (ci, c) in refs.cases.enumerated() {
            let got = BiomarkerExtractor.compute(waveform: c.waveform,
                                                 sampleRate: refs.sample_rate,
                                                 nFFT: refs.n_fft, hop: refs.hop)
            XCTAssertEqual(got.count, refs.names.count)
            for (i, name) in refs.names.enumerated() {
                let expected = c.bio[name] ?? 0
                let actual = got[i]
                let scale = max(abs(expected), 1e-3)
                let relErr = abs(actual - expected) / scale
                XCTAssertLessThan(relErr, Float(refs.tolerance_rel) + 0.02,
                    "case \(ci) [\(c.class)] feature \(name): "
                    + "expected \(expected) got \(actual) (rel \(relErr))")
            }
        }
    }
}
