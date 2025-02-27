from typing import Dict, Any
from ..models.glycan_encoders.dummy import DummyGlycanEncoder
from ..models.glycan_encoders.chemberta import ChemBERTaEncoder
from ..models.protein_encoders.dummy import DummyProteinEncoder
from ..models.protein_encoders.esm3 import ESM3Encoder
from ..models.binding_predictors.dummy import DummyBindingPredictor
from ..models.binding_predictors.dnn import DNNBindingPredictor
from ..models.protein_encoders.esmc import ESMCEncoder
from ..models.glycan_encoders.fingerprint import RDKITFingerprintEncoder


def create_glycan_encoder(encoder_type: str, **kwargs) -> Any:
    """Create glycan encoder instance based on type"""
    encoders = {
        'dummy': DummyGlycanEncoder,
        'chemberta': ChemBERTaEncoder,
        'rdkit-fingerprint': RDKITFingerprintEncoder,
    }
    
    encoder = encoders[encoder_type]
    return encoder(**kwargs)


def create_protein_encoder(encoder_type: str, **kwargs) -> Any:
    encoders = {
        'dummy': DummyProteinEncoder,
        'esm3': ESM3Encoder,
        'esmc': ESMCEncoder,
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
