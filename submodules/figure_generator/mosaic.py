import cv2


class Mosaic:
    @classmethod
    def concat_tile(cls, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    @classmethod
    def get_mosaic(cls,
                   width,
                   height,
                   list_loaded_images):
        assert width * height == len(list_loaded_images)
        l = [[list_loaded_images[i + j * width] for i in range(width)] for j in range(height)]
        return cls.concat_tile(l)

    @classmethod
    def get_mixed_mosaic(cls,
                         width,
                         height,
                         list_loaded_images_1,
                         list_loaded_images_2):
        assert width * height == len(list_loaded_images_1)
        assert width * height == len(list_loaded_images_2)

        l_1 = [[list_loaded_images_1[i + j * width] for i in range(width)] for j in range(height)]
        l_2 = [[list_loaded_images_2[i + j * width] for i in range(width)] for j in range(height)]

        l_mixed = [item
                   for sublist in zip(l_1, l_2)
                   for item in sublist]
        return cls.concat_tile(l_mixed)

    @classmethod
    def save_mosaic(cls,
                    mosaic,
                    path):
        cv2.imwrite(path, mosaic)

    @classmethod
    def get_list_images(cls, path_template, list_gen):
        l = []
        filenames = [path_template.format(gen) for gen in list_gen]
        for name in filenames:
            l.append(cv2.imread(name))
        return l

    @classmethod
    def resize_images(cls, list_loaded_images, new_shape):
        return [cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA) for img in list_loaded_images]

    @classmethod
    def generate_mosaic(cls,
                        path_template,
                        list_gen,
                        width,
                        height,
                        path_save,
                        new_shape=None,
                        ):
        list_images = cls.get_list_images(path_template, list_gen)
        if new_shape:
            list_images = cls.resize_images(list_images, new_shape)
        mosaic = cls.get_mosaic(width, height, list_images)
        cls.save_mosaic(mosaic, path_save)

    @classmethod
    def generate_mixed_mosaic(cls,
                              path_template_1,
                              path_template_2,
                              list_indivs,
                              width,
                              height,
                              path_save,
                              new_shape_1=None,
                              new_shape_2=None,
                              ):
        list_images_1 = cls.get_list_images(path_template_1, list_indivs)
        list_images_2 = cls.get_list_images(path_template_2, list_indivs)
        if new_shape_1:
            list_images_1 = cls.resize_images(list_images_1, new_shape_1)
        if new_shape_2:
            list_images_2 = cls.resize_images(list_images_2, new_shape_2)

        mosaic = cls.get_mixed_mosaic(width, height, list_images_1, list_images_2)
        cls.save_mosaic(mosaic, path_save)


if __name__ == '__main__':
    Mosaic.generate_mosaic(
        path_template="mosaic_load/reconstruction_obs_gen_0002000_indiv_00000{}_rgb.png",
        # path_template="mosaic_load/observation_gen_0002000_indiv_00000{}_color.png",
        list_gen=[f"{x:02}" for x in range(84)],
        width=12,
        height=7,
        path_save="mosaic_save/test-reconstruction.png",
        new_shape=(64, 64)
    )

    Mosaic.generate_mixed_mosaic(
        path_template_1="mosaic_load/observation_gen_0002000_indiv_00000{}_color.png",
        path_template_2="mosaic_load/reconstruction_obs_gen_0002000_indiv_00000{}_rgb.png",
        # path_template="mosaic_load/observation_gen_0002000_indiv_00000{}_color.png",
        list_indivs=[f"{x:02}" for x in range(84)],
        width=12,
        height=7,
        path_save="mosaic_save/test-reconstruction.png",
        new_shape_1=(64, 64),
        new_shape_2=(64, 64)
    )
