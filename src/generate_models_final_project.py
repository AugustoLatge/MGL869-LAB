from src.modules.model_final_project_part_1 import generate_final_project_part_1_model
from src.modules.model_final_project_part_2 import generate_final_project_part_2_model

versions = [
    "2_0_0",
    "2_1_0",
    "2_2_0",
    "2_3_0",
    "3_0_0",
    "3_1_0"
]

for version in versions:
    generate_final_project_part_1_model(version)
    generate_final_project_part_2_model(version)
