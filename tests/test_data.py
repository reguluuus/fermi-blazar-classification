import numpy as np
import pandas as pd

from agn_classifier.data import fit_preprocessor


def test_preprocessor_handles_missing_values_and_preserves_shape():
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    preprocessor = fit_preprocessor(train)
    transformed = preprocessor.transform(train)

    assert transformed.shape == (4, 2)
    assert np.isfinite(transformed).all()
    assert preprocessor.feature_names == ["a", "b"]
