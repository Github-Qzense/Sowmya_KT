from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def sardine_model(save_path="generated-results/sardine_model_flowchart.png", dpi=300):
    # Input layer
    input_layer = Input(shape=(224, 224, 3), name="Damage_Detection")

    # Base model
    base_model = DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model._name = "densenet121"

    x = base_model(input_layer)

    # Global Average Pooling
    x = GlobalAveragePooling2D(
        name="global_average_pooling2d_1"
    )(x)

    # Fully connected layers
    x = Dense(512, activation="relu", name="dense")(x)
    x = BatchNormalization(name="batch_normalization")(x)
    x = Dropout(0.1, name="dropout")(x)

    x = Dense(256, activation="relu", name="dense_1")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = Dropout(0.1, name="dropout_1")(x)

    # Output layer
    output_layer = Dense(
        1,
        activation="sigmoid",
        name="dense_2",
    )(x)

    # Model creation
    model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name="Damage_Detection",
    )

    # Save architecture
    plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
        dpi=dpi,
    )


    print(f"✓ Model architecture saved to: {save_path}")
    print(f"✓ DPI: {dpi}")

    return model


def mackerel_model(save_path="generated-results/mackerel_model_flowchart.png", dpi=300):
    # Input layer
    input_layer = Input(shape=(224, 224, 3), name="Damage_Detection")

    # Base model
    base_model = DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model._name = "densenet121"

    x = base_model(input_layer)

    # Global Average Pooling
    x = GlobalAveragePooling2D(
        name="global_average_pooling2d_1"
    )(x)

    # Fully connected layers
    x = Dense(128, activation="relu", name="dense")(x)
    x = BatchNormalization(name="batch_normalization")(x)
    x = Dropout(0.1, name="dropout")(x)

    x = Dense(64, activation="relu", name="dense_1")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = Dropout(0.1, name="dropout_1")(x)

    x = Dense(32, activation="relu", name="dense_2")(x)
    x = BatchNormalization(name="batch_normalization_2")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    
    # Output layer
    output_layer = Dense(
        1,
        activation="sigmoid",
        name="dense_3",
    )(x)

    # Model creation
    model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name="Damage_Detection",
    )

    # Save architecture
    plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
        dpi=dpi,
    )

    print(f"✓ Model architecture saved to: {save_path}")
    print(f"✓ DPI: {dpi}")

    return model

if __name__ == "__main__":
    sardine_model()
    mackerel_model()