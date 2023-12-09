import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

image_dir = 'train'
labels_path = 'data_train'
labels_df = pd.read_csv(labels_path)
btf_labels = labels_df['label']
num_classes = btf_labels.nunique()

train_df, valid_df = train_test_split(labels_df, test_size=0.2, random_state=42)
train_df['filename'] = train_df['filename'].astype(str)
valid_df['filename'] = valid_df['filename'].astype(str)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3
)
train_df['label'] = train_df['label'].astype(str)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=image_dir,
                                                    x_col="filename",
                                                    y_col="label",
                                                    target_size=(128, 128),
                                                    batch_size=32,
                                                    save_to_dir='/content/',
                                                    class_mode='categorical')