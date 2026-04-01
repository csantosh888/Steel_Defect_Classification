# Dataset

## Data Preparation

Images are downloaded from the `Kaggle` competition `Severstal Steel Defect Detection`. It has two folders, Test and Train. The `train.csv` file contains the information on the some of the image files classified into four types of defects. This file is used to create folders with images of four defect types. For simplicity, handful amount of images belonging to each group are randomly selected and also a folder with images of no defect is created. The folder `test_images` contains all the images without no defect classification.

Total images : 430 (for train, validation and test)
```
data/steel_defect/
├── no_defect/    ← Class 0 (120 Images)
├── defect_1/     ← Class 1 (90 Images)
├── defect_2/     ← Class 2 (20 Images)
├── defect_3/     ← Class 3 (150 Images)
├── defect_4/     ← Class 4 (50 Images)
└── test_images/  ← (5506 Images)

```

## Data Preprocessing

The most important step in the data preprocessing is `resizing`. The original images of steel has size width of 1600 and height of 256. Here, I did the resizing to the width of 800 and height of 128 to preserve the aspect ratio of original image.




