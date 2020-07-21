from pie_data_zhe import PIE
import pickle


### extracted the annotation data files. ###
# pie_path = '.'
# imdb = PIE(data_path=pie_path)
# imdb.generate_database()

### print data structure ###
# with open('data_cache/pie_database.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data.keys()) # dict_keys(['set01', 'set02', 'set03', 'set04', 'set05', 'set06'])
# print(data['set01'].keys()) # dict_keys(['video_0001', 'video_0002', 'video_0003', 'video_0004'])
# print(data['set01']['video_0001'].keys()) # dict_keys(['num_frames', 'width', 'height', 'ped_annotations', 'traffic_annotations', 'vehicle_annotations'])
# print(data['set01']['video_0001']['ped_annotations'].keys()) # dict_keys(['1_1_1', '1_1_2', '1_1_3', '1_1_4', '1_1_5', '1_1_6', '1_1_7', '1_1_9', '1_1_13', '1_1_20', '1_1_21', '1_1_8', '1_1_14', '1_1_15', '1_1_18', '1_1_19', '1_1_16', '1_1_17', '1_1_10', '1_1_12', '1_1_11'])
# print(data['set01']['video_0001']['ped_annotations']['1_1_1'].keys()) # dict_keys(['frames', 'bbox', 'occlusion', 'behavior', 'attributes'])
# print(data['set01']['video_0001']['ped_annotations']['1_1_1']['frames'])
# print(data['set01']['video_0001']['ped_annotations']['1_1_3']['behavior'])

### pedestrian-focused matching annotations and figures ###
pie_path = '.'
imdb = PIE(data_path=pie_path)
with open('data_cache/pie_database.pkl', 'rb') as f:
    data = pickle.load(f)
ped_data = data['set01']['video_0001']['ped_annotations']['1_1_3']
ped_frames = ped_data['frames']
ped_bboxes = ped_data['bbox']
print(len(ped_frames))
print(len(ped_bboxes))
print(ped_bboxes)
imdb.extract_and_save_images_pedestrian(ped_frames, ped_bboxes)