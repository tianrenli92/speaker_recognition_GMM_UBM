from os.path import dirname,abspath,join
from voice_activity_detection import create_dataset
TEST_AUDIO_FOLDER = join(dirname(dirname(abspath(__file__))), 'data', 'testwav','0713')
test_df = create_dataset(TEST_AUDIO_FOLDER)