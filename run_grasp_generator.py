from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id= '247122070300',
        saved_model_path='/home/leitrechner/CAPS-Framework-/trained-models/epoch_08_iou_1.00',
        visualize=True
    )
    generator.load_model()
    generator.run()
