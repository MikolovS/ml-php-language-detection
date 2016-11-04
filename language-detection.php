<?php

require_once('vendor/autoload.php');

use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WordTokenizer;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\Metric\Accuracy;
use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;

$vectorizer = new TokenCountVectorizer(new WordTokenizer());
$tfIdfTransformer = new TfIdfTransformer();

$dataset = json_decode(file_get_contents("data/training-languages.json"), true);

$samples = [];
$targets = [];
foreach ($dataset as $sample) {
    $samples[] = $sample['sentence'];
    $targets[] = $sample['language'];
}

$vectorizer->fit($samples);
$vectorizer->transform($samples);

$tfIdfTransformer->fit($samples);
$tfIdfTransformer->transform($samples);

$dataset = new ArrayDataset($samples, $targets);

$randomSplit = new StratifiedRandomSplit($dataset, .8);

$classifier = new SVC(Kernel::RBF, 10000);
$classifier->train($randomSplit->getTrainSamples(), $randomSplit->getTrainLabels());

$predictedLabels = $classifier->predict($randomSplit->getTestSamples());

$testLabels = $randomSplit->getTestLabels();
for ($i=0;$i<count($testLabels);$i++) {
    if ($testLabels[$i] == $predictedLabels[$i]) {
        echo $i . ' pass' . PHP_EOL;
    } else {
        echo $i . ' fail' . PHP_EOL;
    }
}

echo 'Accuracy: ' . (Accuracy::score($randomSplit->getTestLabels(), $predictedLabels)) * 100 . '%' . PHP_EOL;