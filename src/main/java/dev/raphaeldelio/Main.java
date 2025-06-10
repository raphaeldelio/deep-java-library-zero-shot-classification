package dev.raphaeldelio;

import ai.djl.MalformedModelException;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.translator.ZeroShotClassificationInput;
import ai.djl.modality.nlp.translator.ZeroShotClassificationOutput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws Exception {
        /* facebook/bart-large-mnlu */
        var bartLargeMnli = getCriteriaFromHub(
                "djl://ai.djl.huggingface.pytorch/facebook/bart-large-mnli",
                "model/bart-large-mnli/tokenizer.json"
        );

        var resultMultiLabel = runPrediction(bartLargeMnli, true);
        var scoreMultiLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.8831513524055481,
                "Software Engineering", 0.5423187017440796,
                "Politics", 6.025559268891811E-4
        ));
        if (!resultMultiLabel.equals(scoreMultiLabel)) {
            throw new RuntimeException("Results do not match: " + resultMultiLabel + ", " + scoreMultiLabel);
        }

        var resultSingleLabel = runPrediction(bartLargeMnli, false);
        var scoreSingleLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.8297517895698547,
                "Software Engineering", 0.15263374149799347,
                "Politics", 0.017614545300602913
        ));
        if (!resultSingleLabel.equals(scoreSingleLabel)) {
            throw new RuntimeException("Results do not match: " + resultSingleLabel + ", " + scoreSingleLabel);
        }

        /* DeBERTa-v3-large-mnli-fever-anli-ling-wanli */
        var deberta = getCriteriaFromLocal(
                "model/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                "tokenizer.json"
        );

        resultMultiLabel = runPrediction(deberta, true);
        scoreMultiLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.9982864260673523,
                "Software Engineering", 0.7510337233543396,
                "Politics", 0.00020543170103337616
        ));
        if (!resultMultiLabel.equals(scoreMultiLabel)) {
            throw new RuntimeException("Results do not match: " + resultMultiLabel + ", " + scoreMultiLabel);
        }

        resultSingleLabel = runPrediction(deberta, false);
        scoreSingleLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.9766120314598083,
                "Software Engineering", 0.021817317232489586,
                "Politics", 0.0015706506092101336
        ));
        if (!resultSingleLabel.equals(scoreSingleLabel)) {
            throw new RuntimeException("Results do not match: " + resultSingleLabel + ", " + scoreSingleLabel);
        }

        /* tasksource/ModernBERT-base-nli */
        var modern =  getCriteriaFromLocal(
                "model/ModernBERT-base-nli",
                "tokenizer.json"
        );

        resultMultiLabel = runPrediction(modern, true);
        scoreMultiLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.9858733415603638,
                "Software Engineering", 0.605042040348053,
                "Politics", 0.0009751664474606514
        ));
        if (!resultMultiLabel.equals(scoreMultiLabel)) {
            throw new RuntimeException("Results do not match: " + resultMultiLabel + ", " + scoreMultiLabel);
        }

        resultSingleLabel = runPrediction(modern, false);
        scoreSingleLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.9219518899917603,
                "Software Engineering", 0.07191568613052368,
                "Politics", 0.006132523063570261
        ));
        if (!resultSingleLabel.equals(scoreSingleLabel)) {
            throw new RuntimeException("Results do not match: " + resultSingleLabel + ", " + scoreSingleLabel);
        }

        /* MoritzLaurer/bge-m3-zeroshot-v2.0 */
        var bge = getCriteriaFromLocal(
                "model/bge-m3-zeroshot-v2.0",
                "tokenizer.json"
        );

        resultMultiLabel = runPrediction(bge, true);
        scoreMultiLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.9832621812820435,
                "Software Engineering", 0.06830243021249771,
                "Politics", 0.000334366864990443
        ));
        if (!resultMultiLabel.equals(scoreMultiLabel)) {
            throw new RuntimeException("Results do not match: " + resultMultiLabel + ", " + scoreMultiLabel);
        }

        resultSingleLabel = runPrediction(bge, false);
        scoreSingleLabel = new LinkedHashMap<>(Map.of(
                "Software Programming", 0.953702449798584,
                "Software Engineering", 0.0426984541118145,
                "Politics", 0.0035991526674479246
        ));
        if (!resultSingleLabel.equals(scoreSingleLabel)) {
            throw new RuntimeException("Results do not match: " + resultSingleLabel + ", " + scoreSingleLabel);
        }
    }

    private static Map<String, Double> runPrediction(
            Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> criteria,
            boolean multiLabel
    ) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {

        try (ZooModel<ZeroShotClassificationInput, ZeroShotClassificationOutput> model = ModelZoo.loadModel(criteria);
             Predictor<ZeroShotClassificationInput, ZeroShotClassificationOutput> predictor = model.newPredictor()) {

            String inputText = "Java is the best programming language";
            String[] candidateLabels = {"Software Engineering", "Software Programming", "Politics"};
            ZeroShotClassificationInput input = new ZeroShotClassificationInput(inputText, candidateLabels, multiLabel);

            ZeroShotClassificationOutput result = predictor.predict(input);

            Map<String, Double> output = new LinkedHashMap<>();
            String[] labels = result.getLabels();
            double[] scores = result.getScores();
            for (int i = 0; i < labels.length; i++) {
                output.put(labels[i], scores[i]);
            }

            System.out.println("\n[" + criteria.getModelName() + "] Classification results:");
            output.forEach((label, score) -> System.out.printf("%s: %.6f%n", label, score));

            return output;
        }
    }

    private static Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> getCriteriaFromHub(
            String modelUrl, String tokenizerPath
    ) throws IOException {
        var tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerPath));
        var translator = ZeroShotClassificationTranslator.builder(tokenizer).build();

        return Criteria.builder()
                .optModelUrls(modelUrl)
                .optEngine("PyTorch")
                .setTypes(ZeroShotClassificationInput.class, ZeroShotClassificationOutput.class)
                .optTranslator(translator)
                .build();
    }

    private static Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> getCriteriaFromLocal(
            String modelDir, String tokenizerFile
    ) throws IOException {
        Path tokenizerPath = Paths.get(modelDir).resolve(tokenizerFile);
        var tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);
        var translator = ZeroShotClassificationTranslator.builder(tokenizer)
                .build();

        return Criteria.builder()
                .optModelPath(Paths.get(modelDir))
                .optEngine("PyTorch")
                .setTypes(ZeroShotClassificationInput.class, ZeroShotClassificationOutput.class)
                .optTranslator(translator)
                .build();
    }
}