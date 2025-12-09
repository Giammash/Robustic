from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Optional, Dict, Union, Literal

if TYPE_CHECKING:
    pass


# TODO: Implement the Dataset class for your model
class Dataset:
    def __init__(self, data: Dict) -> None:
        """
        Initialize the Dataset class.

        Parameters
        ----------
        data : Dict[str, Dict[Literal["emg", "kinematics"], np.ndarray]]
            The recordings selected for creating the dataset.
            The data is stored in a dictionary with the following structure:
            {
                "Task selected (e.g. 'Rest' or 'Fist')": {
                    "emg": np.ndarray,
                    "kinematics": np.ndarray,
                }
            }
            The emg np.ndarray is a 2D array with the shape (32, recording_time * 2000 Hz).
            The kinematics np.ndarray is a 2D array with the shape (9, recording_time * 60 Hz).
        """
        self.data = data

    def create_dataset(
        self,
    ) -> Dict[Literal["training", "testing"], Dict[Literal["x", "y"], Any]]:
        """
        Create the dataset for the model.
        """

        # TODO: Implement the dataset creation
        # TODO: If doing regression, the y is the kinematics otherwise it is the task label
        # TODO: If using kinematics, make sure to oversample the data to match the number of samples in the emg

        return {
            "training": {
                "x": None,
                "y": None,
            },
            "testing": {
                "x": None,
                "y": None,
            },
        }
