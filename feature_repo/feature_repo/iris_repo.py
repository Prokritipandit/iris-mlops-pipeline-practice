#from google.protobuf.duration_pb2 import Duration
from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32

# Define the Parquet file we just created as the source
iris_feature_source = FileSource(
    path="../../data/iris_features.parquet",  # Path is relative to the feature_repo/ directory
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define our entity (the "primary key" of our features)
iris_entity = Entity(
    name="iris_id", 
    value_type=ValueType.INT64, 
    description="The ID of the iris plant"
)

# Define the Feature View
# This groups our features together and links them to the source
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=365),  # How long to keep features in the online store
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,  # We want this available in the online store
    source=iris_feature_source,
    tags={"dataset": "iris"},
)