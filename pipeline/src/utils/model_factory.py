from typing import Dict, Any
from ..models.glycan_encoders.dummy import DummyGlycanEncoder
from ..models.glycan_encoders.chemberta import ChemBERTaEncoder
from ..models.protein_encoders.dummy import DummyProteinEncoder
from ..models.binding_predictors.dummy import DummyBindingPredictor
from ..models.binding_predictors.dnn import DNNBindingPredictor
from ..models.protein_encoders.esmc import ESMCEncoder
from ..models.protein_encoders.ankh import AnkhEncoder
from ..models.protein_encoders.prostT5 import ProstT5Encoder
from ..models.protein_encoders.feature import FeatureBasedProteinEncoder


def create_glycan_encoder(encoder_type: str, **kwargs) -> Any:
    """Create glycan encoder instance based on type"""
    encoders = {
        'dummy': DummyGlycanEncoder,
        'chemberta': ChemBERTaEncoder,
    }
    
    encoder = encoders[encoder_type]
    return encoder(**kwargs)


def create_protein_encoder(encoder_type: str, **kwargs) -> Any:
    encoders = {
        'dummy': DummyProteinEncoder,
        'esmc': ESMCEncoder,
        'ankh': AnkhEncoder,
        'prostt5': ProstT5Encoder,
        'feature': FeatureBasedProteinEncoder,
    }

    encoder = encoders[encoder_type]
    return encoder(**kwargs)
    
    
def create_binding_predictor(predictor_type: str, **kwargs) -> Any:
    predictors = {
        'dummy': DummyBindingPredictor,
        'dnn': DNNBindingPredictor
    }
    
    predictor = predictors[predictor_type]
    return predictor(**kwargs)
