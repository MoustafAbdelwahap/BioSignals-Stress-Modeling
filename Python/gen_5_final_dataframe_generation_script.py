# script to transform the raw Dataset into a Pandas Dataframe

# import my_paths
from gen_1_physiological_features_computing import *
from gen_2_physiological_features_merging import *
from gen_3_meteorological_data_treatment import *
from gen_4_hbh_data_merging import *

# Names and paths
raw_dataset_path = 'C:/Users/yanni/Desktop/MIAMS/MIAMS_raw'
features_dataset_path = 'C:/Users/yanni/Desktop/MIAMS/MIAMS_features'

## 1. Copy the original dataset architecture
my_paths.copy_folder_architecture(
    original_directory_path=raw_dataset_path,
    target_directory_path=features_dataset_path
)

# 2. Physiological data descriptors calculation
computing_parameters = Computing_Parameters(
        window_length=3600, 
        window_step_size=3600,
        hrv_threshold=0.1,
        hrv_clean_data=False,
        only_on_full_hour_slots=True
)
compute_features_for_all_dataset_sessions(
    raw_dataset_path=raw_dataset_path,
    destination_dataset_path=features_dataset_path,
    computing_parameters=computing_parameters
)

# 3. Physiological data descriptors merge

# merge by participant
merge_features_by_participant_for_all(
    dataset_path=features_dataset_path,
    computing_parameters=computing_parameters
)

# merge all
merge_all_participants_features_in_a_dataframe(
    dataset_path=features_dataset_path,
    computing_parameters=computing_parameters
)

# 4. Processing of meteorological data
group_meteorological_data_for_Dijon_from_raw_csv_files(
    raw_weather_csv_folder_path = my_paths.get_dataset_weather_directory_path(dataset_path=raw_dataset_path),
    destination_directory_path = my_paths.get_dataset_weather_directory_path(dataset_path=features_dataset_path),
)

# 5. Final Merge
merge_all(
    computed_dataset_path=computed_dataset_path,
    computing_parameters=computing_parameters
)
