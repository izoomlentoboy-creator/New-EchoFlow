import CoreML
import Foundation

/// High-level on-device API. Feeds the raw waveform + clinical biomarkers to
/// the Core ML model and returns a calibrated, hierarchical verdict that
/// mirrors `echoflow/predict.py` (including abstention for safety).
///
///   let analyzer = try VoiceAnalyzer()
///   let result = try analyzer.analyze(waveform: waveform)
///   print(result.summary)
struct VoiceResult {
    let probabilities: [String: Double]   // class -> probability (sums to 1)
    let topClass: String
    let confidence: Double
    let pathologyProbability: Double
    let severity: Double
    let voicedFraction: Double
    let isInconclusive: Bool
    let summary: String
}

final class VoiceAnalyzer {

    // Must match echoflow.config.CLASSES (order matters!)
    static let classes = ["healthy", "hyperfunctional", "paralysis",
                          "neurological", "inflammatory"]
    static let minConfidence = 0.45
    static let minVoicedFraction = 0.30

    private let model: MLModel

    init(modelURL: URL) throws {
        self.model = try MLModel(contentsOf: modelURL)
    }

    /// Convenience init for a model bundled as "EchoFlow.mlmodelc".
    convenience init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: "EchoFlow", withExtension: "mlmodelc") else {
            throw NSError(domain: "EchoFlow", code: 404,
                          userInfo: [NSLocalizedDescriptionKey: "EchoFlow.mlmodelc not found"])
        }
        try self.init(modelURL: url)
    }

    func analyze(waveform: [Float]) throws -> VoiceResult {
        let bio = BiomarkerExtractor.compute(waveform: waveform,
                                             sampleRate: Int(AudioRecorder.sampleRate))
        let voicedFraction = Double(bio[BiomarkerExtractor.voicedFractionIndex])

        let wavArr = try MLMultiArray(shape: [1, NSNumber(value: waveform.count)],
                                      dataType: .float32)
        for (i, v) in waveform.enumerated() { wavArr[i] = NSNumber(value: v) }
        let bioArr = try MLMultiArray(shape: [1, NSNumber(value: bio.count)],
                                      dataType: .float32)
        for (i, v) in bio.enumerated() { bioArr[i] = NSNumber(value: v) }

        let input = try MLDictionaryFeatureProvider(features: [
            "waveform": MLFeatureValue(multiArray: wavArr),
            "bio": MLFeatureValue(multiArray: bioArr),
        ])
        let out = try model.prediction(from: input)

        // Read probabilities: prefer the classifier dict, fall back to "probs".
        var probs = [String: Double]()
        if let dict = out.featureValue(for: "classProbs")?.dictionaryValue as? [String: Double] {
            probs = dict
        } else if let arr = out.featureValue(for: "probs")?.multiArrayValue {
            for (i, c) in Self.classes.enumerated() {
                probs[c] = arr[i].doubleValue
            }
        }
        let severity = out.featureValue(for: "severity")?.multiArrayValue?[0].doubleValue ?? 0

        let top = probs.max { $0.value < $1.value }
        let topClass = top?.key ?? "healthy"
        let confidence = top?.value ?? 0
        let pathologyProb = 1.0 - (probs["healthy"] ?? 0)

        let inconclusive = voicedFraction < Self.minVoicedFraction
                        || confidence < Self.minConfidence
        let summary = Self.makeSummary(topClass: topClass, confidence: confidence,
                                       pathologyProb: pathologyProb, severity: severity,
                                       voicedFraction: voicedFraction,
                                       inconclusive: inconclusive)
        return VoiceResult(probabilities: probs, topClass: topClass,
                           confidence: confidence, pathologyProbability: pathologyProb,
                           severity: severity, voicedFraction: voicedFraction,
                           isInconclusive: inconclusive, summary: summary)
    }

    private static func makeSummary(topClass: String, confidence: Double,
                                    pathologyProb: Double, severity: Double,
                                    voicedFraction: Double, inconclusive: Bool) -> String {
        if voicedFraction < minVoicedFraction {
            return "Неинформативно: мало голоса. Запишите гласный /а/ 3–5 секунд."
        }
        if inconclusive {
            return "Неинформативно (уверенность \(Int(confidence*100))%). "
                 + "Рекомендуется консультация специалиста."
        }
        if topClass == "healthy" {
            return "Голос здоров (\(Int(confidence*100))%)."
        }
        let band = severity < 0.33 ? "лёгкая" : (severity < 0.66 ? "умеренная" : "выраженная")
        return "Вероятна патология (\(Int(pathologyProb*100))%). "
             + "Тип: \(topClass) (\(Int(confidence*100))%). Тяжесть: \(band)."
    }
}
