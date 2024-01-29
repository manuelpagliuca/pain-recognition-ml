from utility import FeatureExtraction
from utility import DataGeneration
from utility import TestingModels

if __name__ == '__main__':
    # Feature engineering (biosignals, video, early fusion, binary biosignals+video+ef)
    FeatureExtraction.extract_bio_features()
    FeatureExtraction.extract_video_features()

    # Data generation
    DataGeneration.binarize_biosignals_and_video_features()
    DataGeneration.early_fusion()
    DataGeneration.binarize_early_fusion_features()

    # Testing Models
    TestingModels.binary_classification()
    TestingModels.multiclass_classification()
