# PeopleAtPlaces Dataset

Release of the additional annotation files and images of the People@Places Dataset. The dataset is derived from the [Places365-Standard dataset](http://places2.csail.mit.edu/download.html). 

An overview of the dataset annotation process is shown in the image below.

![Dataset creation diagram](dataset_creation.png?raw=true "Data creation")


## Annotations

The annotations contain per image annotations for bustle (i.e., more or less populated) and shot type (from extreme close-up to extreme long shot) for the training and validation set of Places365-Standard.

### Bustle class definitions

| ID | Class | Definition |
| ----------- | ----------- | ----------- |
| 0 | unpopulated | no persons or vehicles |
| 1 | few people | less than 3 persons, no vehicles, area less than 10% |
| 2 | few vehicles | less than 3 vehicles, no persons, area less than 20% |
| 3 | few large	| less than 3, any area |
| 4 | medium | less than 11 people/vehicles, area less than 30% |
| 5 | populated	| more people/vehicles or covering larger area |
| 999 | undefined | |

### Shot type class definitions

| ID | Class | Definition |
| ----------- | ----------- | ----------- |
| 0 | undefined | |
| 1 | extreme close-up | detail of face |
| 2 | close-up | head |
| 3 | medium close-up | cut under chest |
| 4 | tight medium shot | cut under waist |
| 5 | medium shot | cut under crotch |
| 6 | medium full shot | cut under knee |
| 7 | full shot | person fully visible |
| 8 | long shot | person ~1/3 of frame height |
| 9 | extreme long shot & person | <1/3 of frame height |

The annotations are contained in `annotations/train` and `annotations/val` directories.

`popshotcls.csv` contains all bustle and shot type annotations per image, before augmenting the extreme close-up images.

`popshotcls_ecu.csv` contains all bustle and shot type annotations per image, including the augmented extreme close-up images.

For the validation dataset, `popshotcls_cvat.csv` provides the subset of annotations input to annotation tool, and `popshotcls_checked.csv` contains the annotations after manual checking.

`imagelist_*.txt` contain the list of images actually used in our experiments.

## Raw annotations

This directory contains outputs of intermediate stages of the annotations toolchain.

## Images

This directory contains the images created by cropping additional extreme close-up images in `*_ecu`. 

## License

The license terms of the [Places365 dataset](http://places2.csail.mit.edu/download.html) apply to the images. The additional annotates provided here are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).