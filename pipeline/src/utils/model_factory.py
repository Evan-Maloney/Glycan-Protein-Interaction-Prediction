from typing import Any

from ..models.glycan_encoders.dummy import DummyGlycanEncoder
from ..models.glycan_encoders.chemberta import ChemBERTaEncoder
from ..models.glycan_encoders.rdkit import RDKITGlycanEncoder
from ..models.glycan_encoders.gnn import GNNGlycanEncoder
from ..models.glycan_encoders.aconn_gnn import ACONNEncoder
from ..models.glycan_encoders.aconn_gnnV2 import ACONNEncoderV2
from ..models.glycan_encoders.sweet_talk import SweetTalkGlycanEncoder
from ..models.glycan_encoders.sweetnet import SweetNetEncoder
from ..models.glycan_encoders.mpnn import MPNNGlycanEncoder
from ..models.glycan_encoders.gifflar import GIFFLAREncoder

from ..models.protein_encoders.dummy import DummyProteinEncoder
from ..models.binding_predictors.dummy import DummyBindingPredictor
from ..models.protein_encoders.biopy import BioPyProteinEncoder

from ..models.binding_predictors.dnn import DNNBindingPredictor
from ..models.protein_encoders.esmc import ESMCEncoder

from ..models.binding_predictors.attention import AttentionFusionBindingPredictor

from ..models.protein_encoders.pt_gnn import AdvancedGNNProteinEncoder
from ..models.protein_encoders.lstm import LSTMProteinEncoder

from ..models.binding_predictors.mean_predictor import MeanPredictor
from ..models.binding_predictors.zero_predictor import ZeroPredictor


from ..models.protein_encoders.biopy import BioPyProteinEncoder
from ..models.protein_encoders.esm2_320 import ESM2Encoder



def create_glycan_encoder(encoder_type: str, **kwargs) -> Any:
    """Create glycan encoder instance based on type"""
    encoders = {
        'dummy': DummyGlycanEncoder,
        'chemberta': ChemBERTaEncoder,
        'rdkit': RDKITGlycanEncoder,
        'gnn': GNNGlycanEncoder,
        'aconn_gnn': ACONNEncoder,
        'aconn_gnnV2': ACONNEncoderV2,
        'sweettalk': SweetTalkGlycanEncoder,
        'sweetnet': SweetNetEncoder,
        'mpnn': MPNNGlycanEncoder,
        'gifflar': GIFFLAREncoder
    }
    
    encoder = encoders[encoder_type]
    return encoder(**kwargs)


def create_protein_encoder(encoder_type: str, **kwargs) -> Any:
    encoders = {
        'dummy': DummyProteinEncoder,
        'esmc': ESMCEncoder,
        'biopy': BioPyProteinEncoder,
        'pt_gnn': AdvancedGNNProteinEncoder,
        'lstm': LSTMProteinEncoder,
        'esm2': ESM2Encoder
    }

    encoder = encoders[encoder_type]
    return encoder(**kwargs)
    
    
def create_binding_predictor(predictor_type: str, **kwargs) -> Any:
    predictors = {
        'dummy': DummyBindingPredictor,
        'dnn': DNNBindingPredictor,
        'mean': MeanPredictor,
        'attention': AttentionFusionBindingPredictor,
        'zero': ZeroPredictor
    }
    
    predictor = predictors[predictor_type]
    return predictor(**kwargs)
