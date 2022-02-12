import numpy as np
from torch import Tensor
from typing import Tuple, Union, Callable
from . import functional as F

# base element_type
Base_Element_Type = Union[np.ndarray, Tensor]
Image_InOutput_Type = Base_Element_Type  # image
Landmarks_InOutput_Type = Base_Element_Type  # landmarks


class AutoDtypeEnum:
    # autodtype modes
    Array_In: int = 0
    Array_InOut: int = 1
    Tensor_In: int = 2
    Tensor_InOut: int = 3


def autodtype(mode: int) -> Callable:
    # A Pythonic style to auto convert input dtype and let the output dtype unchanged

    assert 0 <= mode <= 3

    def wrapper(
            callable_array_or_tensor_func: Callable
    ) -> Callable:

        def apply(
                self,
                img: Image_InOutput_Type,
                landmarks: Landmarks_InOutput_Type,
                **kwargs
        ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:
            # Type checks
            assert all(
                [isinstance(_, (np.ndarray, Tensor))
                 for _ in (img, landmarks)]
            ), "Error dtype, must be np.ndarray or Tensor!"
            # force array before transform and then wrap back.
            if mode == AutoDtypeEnum.Array_InOut:
                if any((
                        isinstance(img, Tensor),
                        isinstance(landmarks, Tensor)
                )):
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
            # force array before transform and don't wrap back.
            elif mode == AutoDtypeEnum.Array_In:
                if any((
                        isinstance(img, Tensor),
                        isinstance(landmarks, Tensor)
                )):
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            # force tensor before transform and then wrap back.
            elif mode == AutoDtypeEnum.Tensor_InOut:
                if any((
                        isinstance(img, np.ndarray),
                        isinstance(landmarks, np.ndarray)
                )):
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
            # force tensor before transform and don't wrap back.
            elif mode == AutoDtypeEnum.Tensor_In:
                if any((
                        isinstance(img, np.ndarray),
                        isinstance(landmarks, np.ndarray)
                )):
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            else:
                img, landmarks = callable_array_or_tensor_func(
                    self,
                    img,
                    landmarks,
                    **kwargs
                )

            return img, landmarks

        return apply

    return wrapper
