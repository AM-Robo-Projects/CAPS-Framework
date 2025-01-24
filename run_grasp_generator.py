from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id= '032622073169',
        saved_model_path='/home/abdelrahman/ws_moveit/src/robotic-grasping-CNN/trained-models/epoch_08_iou_1.00',
        visualize=True
    )
    generator.load_model()
    generator.run()
