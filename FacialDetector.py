from imports import *
from evals import *
from utils import *
from generator import *
from params import *


class FacialDetector:
    def __init__(self) -> None:
        self.best_model = None

    def get_positive_descriptors(self):

        positive_descriptors = []

        images_path = os.path.join(POSITIVE, '*.jpg')
        files = glob.glob(images_path)

        print(f"Sunt procesate {len(files)} imagini pozitive...")
        for i, image_file in enumerate(files):
            print(f"Imagine procesata: {i + 1}/{len(files)}")
            image = cv2.imread(image_file)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hog_image = hog(gray_image, pixels_per_cell=(6, 6), cells_per_block=(
                2, 2), feature_vector=True)  # pixels_per_cell = 6, 6, cells_per_block = 2, 2
            positive_descriptors.append(hog_image)

            hog_image = hog(np.fliplr(gray_image), pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(hog_image)

            angle = 45
            rotated_image = rotate(
                gray_image, angle, reshape=False, mode='nearest')
            hog_image_rotated = hog(rotated_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(hog_image_rotated)

            jittered_image = exposure.adjust_gamma(
                image, gamma=np.random.uniform(0.5, 1.5))
            hog_image_jittered = hog(cv2.cvtColor(jittered_image, cv2.COLOR_BGR2GRAY), pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(hog_image_jittered)

            sheared_transformer = transform.AffineTransform(
                shear=np.deg2rad(np.random.uniform(-20, 20)))
            sheared_image = transform.warp(gray_image, sheared_transformer)
            hog_image_sheared = hog(sheared_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(hog_image_sheared)

            blurred_image = util.random_noise(
                gray_image, var=np.random.uniform(0.01, 0.1))
            hog_image_blurred = hog(blurred_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(hog_image_blurred)

        print("Descriptorii pozitivi au fost extrasi.")

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        negative_descriptors = []

        images_path = os.path.join(NEGATIVE, '*.jpg')
        files = glob.glob(images_path)

        print(f"Sunt procesate {len(files)} imagini negative...")

        for i, image_file in enumerate(files):
            print(
                f"Imagine procesata: {i + 1}/{len(files)}")
            image = cv2.imread(image_file)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hog_image = hog(gray_image, pixels_per_cell=(6, 6),
                            cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image)

            hog_image = hog(np.fliplr(gray_image), pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image)

            angle = 45
            rotated_image = rotate(
                gray_image, angle, reshape=False, mode='nearest')
            hog_image_rotated = hog(rotated_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image_rotated)

            jittered_image = exposure.adjust_gamma(
                image, gamma=np.random.uniform(0.5, 1.5))
            hog_image_jittered = hog(cv2.cvtColor(jittered_image, cv2.COLOR_BGR2GRAY), pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image_jittered)

            sheared_transformer = transform.AffineTransform(
                shear=np.deg2rad(np.random.uniform(-20, 20)))
            sheared_image = transform.warp(gray_image, sheared_transformer)
            hog_image_sheared = hog(sheared_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image_sheared)

            blurred_image = util.random_noise(
                gray_image, var=np.random.uniform(0.01, 0.1))
            hog_image_blurred = hog(blurred_image, pixels_per_cell=(
                6, 6), cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(hog_image_blurred)

        print("Descriptorii negativi au fost extrasi.")
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels, ignore_restore=True):
        svm_file_name = os.path.join(MODELS, 'model_detector_facial')
        if os.path.exists(svm_file_name) and ignore_restore:
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy, best_c, best_model = 0, 0, None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2]
        for c in Cs:
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]
        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr. exemple antrenare')
        plt.ylabel('Scor clasificator')
        plt.title(
            'Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive',
                   '0', 'Scoruri exemple negative'])
        plt.show()

    def run(self):
        test_images_path = os.path.join(TEST, '*.jpg')
        threshold = -1
        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        file_names = np.array([])
        num_test_images = len(test_files)

        original_window_size = (64, 64)
        pixels_per_cell = (6, 6)
        cells_per_block = (2, 2)
        orientations = 9
        features_per_block = np.prod(cells_per_block) * orientations
        blocks_per_window = (
            original_window_size[0] // pixels_per_cell[0] -
            cells_per_block[0] + 1,
            original_window_size[1] // pixels_per_cell[1] -
            cells_per_block[1] + 1
        )

        desired_num_features = blocks_per_window[0] * \
            blocks_per_window[1] * features_per_block

        for i, image_path in enumerate(test_files):
            print('Imagine procesata %d/%d..' % (i, num_test_images))
            img = cv2.imread(image_path)
            num_rows, num_cols = img.shape[0], img.shape[1]

            original_image = img.copy()

            image_scores = []
            image_detections = []
            current_file_names = []

            scale_factors = [
                (0.8, 1.2), (1.0, 1.5), (1.2, 0.8),
                (0.7, 1.3), (1.5, 1.0),
                (0.9, 1.1), (1.1, 0.9),
                (0.5, 1.5), (1.5, 0.5),
                (1.8, 2.5), (2.0, 3.0),
                (0.4, 2.0), (2.0, 0.4),
                (0.3, 3.0), (3.0, 0.3),
                (0.2, 4.0), (4.0, 0.2),
                (0.1, 5.0), (5.0, 0.1)
            ]
            for scale_factor_w, scale_factor_h in scale_factors:
                scaled_window_size = (
                    int(original_window_size[0] * scale_factor_w),
                    int(original_window_size[1] * scale_factor_h)
                )

                for y in range(0, num_rows - scaled_window_size[1], 8):
                    for x in range(0, num_cols - scaled_window_size[0], 8):
                        window = img[y:y + scaled_window_size[1],
                                     x:x + scaled_window_size[0]]
                        window_copy = window.copy()
                        gray_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

                        resized_gray_window = cv2.resize(
                            gray_window, original_window_size, interpolation=cv2.INTER_AREA)
                        hog_features = hog(resized_gray_window, pixels_per_cell=pixels_per_cell,
                                           cells_per_block=cells_per_block, feature_vector=True)
                        score = self.best_model.decision_function([hog_features])[
                            0]

                        if score >= threshold and check_skin_color_percentage(window_copy) > 0.5 and is_uniform_color(window_copy):
                            x_min, y_min, x_max, y_max = x, y, x + \
                                scaled_window_size[0], y + \
                                scaled_window_size[1]
                            image_detections.append(
                                (x_min, y_min, x_max, y_max))
                            image_scores.append(score)
                            current_file_names.append(image_path)

            if len(image_scores) > 0:
                image_detections, image_scores = non_maximal_suppression(
                    np.array(image_detections), np.array(image_scores), original_image.shape)

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [
                    short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

        return detections, scores, file_names
