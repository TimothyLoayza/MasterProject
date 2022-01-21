"""Image analysis for Master Project.

Created on Tue Sep 28 10:39:00 2021

@author: Timothy
"""

import math
# import csv
from os import listdir
from os.path import isfile, join

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

import imageio
from PIL import Image

import scipy.fft as fourier
# import scipy.misc

import cv2

import ezdxf

import femm


class Image_analysis:
    """Class to do awesome image analysis."""

    def __init__(self):
        self.c = {'max_c': 255  # max value for a color of a pixel
                  }
        self.images = {}
        self.files = {}

        self.default_parameters = {"dir_path": 'data/microscope_cf_images_27_09_2021/27_09_2021_m2',
                                   "um_per_p": 1.075,
                                   "color_threshold": None,
                                   "actual_min": 1,
                                   "target_min": 0.1,
                                   "actual_max": 98,
                                   "target_max": 99.9,
                                   "min_height": 900,
                                   "max_height": 1000,
                                   "min_width": 750,
                                   "max_width": 1750,
                                   "show_threshold": True,
                                   "div_ratio": 7,
                                   "tol_cluster": 2,
                                   "av_thick": 10,
                                   "particle_permeance": 9.3052*10**(-10),
                                   "oil_permeance": 9.3052*10**(-13),
                                   "functions_set": ["crop", "convert_grey", "reverse_color", "color_threshold"]}

# ============================================================================
# STATIC METHODS
# ============================================================================

    @staticmethod
    def f_threshold_histogram(nb_pools, minima):
        """Return a vector for comparison to histogram.

        Parameters
        ----------
        nb_pools : int
            The number of pools which is the dimension of the returning vector.
        minima : float
            Minima inbetween the two waves inbetween [0, 1].

        Returns
        -------
        ret : numpy array of length nb_pools
            Vector for the correlation and finding of the minima.
        """
        x = np.linspace(0, 1, nb_pools)
        div = 0.09
        sig = 0.05
        ret = np.exp(-0.5*((x-minima-div)**2)/(sig)**2) +\
            np.exp(-0.5*((x-minima+div)**2)/(sig)**2)
        ret = ret / np.sum(ret)
        return ret

    @staticmethod
    def f_pic_corr(nb_pools, minima):
        """Return a vector for comparison to histogram.

        Parameters
        ----------
        nb_pools : int
            The number of pools which is the dimension of the returning vector.
        minima : float
            Minima inbetween the two waves inbetween [0, 1].

        Returns
        -------
        ret : numpy array of length nb_pools
            Vector for the correlation and finding of the minima.
        """
        x = np.linspace(0, 1, nb_pools)
        indices = np.array(list(range(nb_pools)))
        lam = 60/256
        r = 0.1*2
        ret = r+1+np.cos(2*math.pi*(x-minima)/lam)
        ret[indices < 256*(minima-lam/2)] = 0
        ret[indices > 256*(minima+lam/2)] = 0
        ret = ret / np.sum(ret)
        return ret

    @staticmethod
    def rebin(a, shape):
        """Resize a 2D array by averaging subarrays.

        Parameters
        ----------
        a : numpy 2D array
            The 2D array which dimension should be reduced.
        shape : list of two elements
            final shape of the 2D array.

        Returns
        -------
        numpy 2D array
            The reduced array.
        """
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    @staticmethod
    def get_circles_distribution(w, e, r, epsilon, nb_circles, rand=True):
        """Return coordinates of the center of the circles for the desired distribution.

        Parameters
        ----------
        w : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        epsilon : TYPE
            DESCRIPTION.
        nb_circles : TYPE
            DESCRIPTION.
        rand : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.
        """
        X = np.zeros(nb_circles)
        Y = np.zeros(nb_circles)
        for circle in range(nb_circles):
            search = True
            while search:
                x0 = r + epsilon + (w-2*(r+epsilon))*np.random.rand()
                y0 = r + epsilon + (e-2*(r+epsilon))*np.random.rand()
                overlap = False
                for i in range(circle):
                    if ((X[i] - x0)**2 + (Y[i] - y0)**2)**0.5 < 2*(r+epsilon):
                        overlap = True
                        break
                if not overlap:
                    search = False
            X[circle] = x0
            Y[circle] = y0
        return (X, Y)

# ============================================================================
# CLASS METHODS
# ============================================================================

# ==== SETTERS ====

    def set_default_parameters(self, default_parameters):
        """Set the default parameters.

        Parameters
        ----------
        default_parameters : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        self.default_parameters = default_parameters

# ==== GETTERS ====

    def get_default_parameters(self):
        """Get the default parameters.

        Returns
        -------
        None.
        """
        return self.default_parameters

# ==== FILE TREATMENT ====

    def get_list_jpg(self, dir_path='data/microscope_cf_images_27_09_2021/27_09_2021_m2'):
        """Get the list of '.jpg' images in a folder.

        Parameters
        ----------
        dir_path : string
            The relative path to the folder.

        Returns
        -------
        None.
        """
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        jpg_files = [f for f in onlyfiles if f[-4:] == '.jpg']
        self.files[dir_path] = jpg_files

    def read_video(self, name='ampro_1%_rotation_1_', um_per_p=0.14286,
                   path="data/Dino_Lite_video/dampro_1%_rotation_1.wmv"):
        """Read a video and extract every frames into images.

        Parameters
        ----------
        name : string
            The name given to the image.
        um_per_p : float
            The real size covered by a pixel.
        path : string
            The relative path to the video.

        Returns
        -------
        None.
        """
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        print(success, type(image))
        count = 0
        while success:
            self.images[name + str(count)] = {}
            self.images[name + str(count)]['data'] = image
            self.images[name + str(count)]['um_per_p'] = um_per_p
            self.images[name + str(count)]['height'] = self.images[name + str(count)]['data'].shape[0]
            self.images[name + str(count)]['width'] = self.images[name + str(count)]['data'].shape[1]
            # cv2.imwrite("frame%d.jpg" % count, image)
            success, image = vidcap.read()
            count += 1

    def read_im(self,
                um_per_p=0.14286,
                name='new_image',
                path="data/microscope_cf_images_27_09_2021/27_09_2021_m2/132_2%_3.jpg"):
        """Read the image and put it into a array of data.

        Parameters
        ----------
        um_per_p : float
            The real size covered by a pixel.
        name : string
            The name given to the image.
        path : string
            The relative path to the image.

        Returns
        -------
        None.
        """
        self.images[name] = {}
        self.images[name]['data'] = imageio.imread(path)
        self.images[name]['um_per_p'] = um_per_p
        self.images[name]['height'] = self.images[name]['data'].shape[0]
        self.images[name]['width'] = self.images[name]['data'].shape[1]

    def read_all_im(self, um_per_p=0.14286):
        """Read all the images in the list of .jpg and put them into a dictionary.

        Parameters
        ----------
        um_per_p : float
            The real size covered by a pixel..

        Returns
        -------
        None.
        """
        for key, files_list in self.files.items():
            for file_name in files_list:
                self.read_im(um_per_p=um_per_p, name=file_name[:-4], path=key + '/' + file_name)

    def write_image(self, name='new_image', save_name='new_image.jpg'):
        """Save the image as an jpg image.

        Parameters
        ----------
        name : string
            The name given to the image.
        save_name : string
            the name of the new image.

        Returns
        -------
        None.
        """
        imageio.imwrite(save_name, self.images[name]['data'])

    def create_dxf(self, name='new_image', cluster_name=None):
        """Write a dxf with the clusters outlines.

        Parameters
        ----------
        name : TYPE, optional
            DESCRIPTION. The default is 'new_image'.
        cluster_name : string
            The name of a specific cluster if only one cluster displayed.

        Returns
        -------
        None.
        """
        if cluster_name:
            doc = ezdxf.new('R2000')
            msp = doc.modelspace()
            msp.add_lwpolyline(self.images[name]['clusters'][cluster_name]['outline'])
            doc.saveas(name + "_" + cluster_name + ".dxf")
        else:
            doc = ezdxf.new('R2000')
            msp = doc.modelspace()
            for key in self.images[name]['clusters']:
                msp.add_lwpolyline(self.images[name]['clusters'][key]['outline'])
            doc.saveas(name + ".dxf")

# ==== GENERAL IMAGE CHANGE ====

    def crop(self, name='new_image', min_height=900, max_height=1000, min_width=750, max_width=1750):
        """Crop the image (by default to remove the scale).

        Parameters
        ----------
        name : string
            The name given to the image.
        min_height : int
            start height of crop.
        max_height : int
            stop height of crop.
        min_width : int
            start width of crop.
        max_width : int
            stop width of crop.

        Returns
        -------
        None.
        """
        self.images[name]['data'] = self.images[name]['data'][min_height:max_height, min_width:max_width, :]
        self.images[name]['height'] = max_height - min_height
        self.images[name]['width'] = max_width - min_width

    def convert_grey(self, name='new_image'):
        """Convert an image to grey scale, take a 3D numpy array and transform it into a 2D numpy array.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        self.images[name]['data'] = ((self.images[name]['data'][:, :, 0].astype(int) +
                                      self.images[name]['data'][:, :, 1].astype(int) +
                                      self.images[name]['data'][:, :, 2].astype(int))/3).astype(np.uint8)

    def reverse_color(self, name='new_image'):
        """Reverse the color, in case the iron appears black in the image.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        self.images[name]['data'] = 255 - self.images[name]['data']

    def color_threshold(self, name='new_image', color_threshold=None):
        """Find the threshold of the pixel value to determine the clusters.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        # Calculate the threshold
        if color_threshold is not None:
            self.images[name]['threshold'] = color_threshold
        else:
            data = self.images[name]['data'].flatten().copy()
            data = np.histogram(data, bins=np.linspace(0, 255+1, 255+1+1))[0]
            corr = np.zeros(128)
            for counter, minima in enumerate(np.linspace(0.40, 0.75, 128)):
                corr[counter] = np.dot(self.f_threshold_histogram(256, minima), data)
            self.images[name]['threshold'] = int(255*(0.4 + 0.35*np.argmax(corr)/128))
        # binarize the image
        data = self.images[name]['data'].copy()
        data[data <= self.images[name]['threshold']] = False
        data[data > self.images[name]['threshold']] = True
        self.images[name]['binary_data'] = data

    def corr_with_curve(self, name='new_image'):
        """Do not work yet.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        data = self.images[name]['data'].flatten().copy()
        data = np.histogram(data, bins=np.linspace(0, 255+1, 255+1+1))[0]
        corr = np.zeros(256)
        for counter, minima in enumerate(np.linspace(0, 1, 256)):
            corr[counter] = np.dot(self.f_pic_corr(256, minima), data/np.sum(data))
        fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
        ax0.plot(corr, c="midnightblue")
        plt.show()
        self.histogram(name)
        # return corr
        # self.images[name]['threshold'] = int(255*(0.4 + 0.35*np.argmax(corr)/128))

    def image_size_change(self, name='new_image', div_ratio=7):
        """Change the size of an image by a defined.

        Parameters
        ----------
        name : string
            The name given to the image.
        div_ratio : float
            the ratio used to reduce the size of the image.

        Returns
        -------
        None.
        """
        im = Image.fromarray(self.images[name]['data'])
        im = im.resize((int(self.images[name]['width']/div_ratio),
                        int(self.images[name]['height']/div_ratio)))
        self.images[name]['data'] = np.array(im)
        self.images[name]['height'] = self.images[name]['data'].shape[0]
        self.images[name]['width'] = self.images[name]['data'].shape[1]
        self.images[name]['um_per_p'] = self.images[name]['um_per_p']*div_ratio

# ==== CLUSTERS RELATED CALCULATIONS ====

    def cluster_identification(self, name='new_image', tol_cluster=2):
        """Identify individually the clusters of iron particles.

        Parameters
        ----------
        name : string
            The name given to the image.
        tol_cluster : int
            The number of pixels to look for the neiboring particle.

        Returns
        -------
        None.
        """
        rows, columns = self.images[name]['binary_data'].nonzero()
        pixels_found = np.zeros(len(rows), dtype=bool)
        pixels_to_search = []
        clusters = {}
        clusters_state = []
        while False in pixels_found:
            if False not in clusters_state:
                current_cluster = "c" + str(len(clusters_state)+1)
                clusters[current_cluster] = {}
                clusters[current_cluster]['position'] = [np.where(pixels_found == False)[0][0]]
                pixels_to_search = [clusters[current_cluster]['position'][0]]
                pixels_found[clusters[current_cluster]['position'][0]] = True
                clusters_state.append(False)
            found = np.array([])
            for pixel in pixels_to_search:
                l_rows = np.where(np.abs(rows-rows[int(pixel)]) <= tol_cluster)[0]
                l_columns = np.where(np.abs(columns-columns[int(pixel)]) <= tol_cluster)[0]
                l_not_found = np.where(pixels_found == False)[0]
                new_found = reduce(np.intersect1d, (l_rows, l_columns, l_not_found))
                found = np.concatenate((found, new_found), axis=None)
                pixels_found[new_found] = True
            pixels_to_search = []
            if len(found) == 0:
                del clusters_state[-1]
                clusters_state.append(True)
            else:
                pixels_to_search = found.copy()
                clusters[current_cluster]['position'] = \
                    np.concatenate((clusters[current_cluster]['position'], found), axis=None)

        for key in clusters:
            clusters[key]['min_row'] = \
                np.amin([rows[int(index)] for index in clusters[key]['position']])
            clusters[key]['max_row'] = \
                np.amax([rows[int(index)] for index in clusters[key]['position']])
            clusters[key]['min_column'] = \
                np.amin([columns[int(index)] for index in clusters[key]['position']])
            clusters[key]['max_column'] = \
                np.amax([columns[int(index)] for index in clusters[key]['position']])
            clusters[key]['rows'] = [rows[int(index)] for index in clusters[key]['position']]
            clusters[key]['columns'] = [columns[int(index)] for index in clusters[key]['position']]
            # clusters[key]['position'] = \
            #     [(rows[int(index)], columns[int(index)]) for index in clusters[key]['position']]
        self.images[name]['clusters'] = clusters

    def calculation_clusters_permeance(self, name='new_image', av_thick=10,
                                       particle_permeance=9.3052*10**(-10),
                                       oil_permeance=9.3052*10**(-13)):
        """Calculate the clusters associated permeance.

        Parameters
        ----------
        name : string
            The name given to the image.
        av_thick : int
            The average thickness of the cluters in terms of number of particles.
        particle_permeance : float
            Mean permeance of one particle. The default is 9.3052*10**(-10).
        oil_permeance : TYPE, optional
            DESCRIPTION. The default is 9.3052*10**(-13).

        Returns
        -------
        None.
        """
        for key in self.images[name]['clusters']:
            min_height = self.images[name]['clusters'][key]['min_row']
            max_height = self.images[name]['clusters'][key]['max_row']
            min_width = self.images[name]['clusters'][key]['min_column']
            max_width = self.images[name]['clusters'][key]['max_column']
            data = np.zeros((self.images[name]['height'], self.images[name]['width']), dtype=bool)
            data[self.images[name]['clusters'][key]['rows'], self.images[name]['clusters'][key]['columns']] = True
            data = data[min_height:max_height+1, min_width:max_width+1]
            nb_particles_per_row = np.sum(data, axis=1)
            permeance_per_row = nb_particles_per_row * av_thick * particle_permeance + \
                ((max_width - min_width + 1) - nb_particles_per_row) * av_thick * oil_permeance
            total_permeance = np.sum(permeance_per_row**(-1))**(-1)
            self.images[name]['clusters'][key]['permeance'] = total_permeance

    def calculation_outline_cluster(self, name='new_image'):
        """Save the cluster as dxf.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        for key in self.images[name]['clusters']:
            min_height = self.images[name]['clusters'][key]['min_row']
            max_height = self.images[name]['clusters'][key]['max_row']
            min_width = self.images[name]['clusters'][key]['min_column']
            max_width = self.images[name]['clusters'][key]['max_column']
            data = np.zeros((self.images[name]['height'], self.images[name]['width']), dtype=bool)
            data[self.images[name]['clusters'][key]['rows'],
                 self.images[name]['clusters'][key]['columns']] = True
            data = data[min_height:max_height+1, min_width:max_width+1]

            outline_left = [(min_width + np.where(row == True)[0][0], min_height + counter)
                            for counter, row in enumerate(data)]
            outline_right = [(min_width + np.where(row == True)[0][-1]+1, min_height + counter)
                             for counter, row in enumerate(data)]
            outline_right = list(reversed(outline_right))
            outline = outline_left + outline_right
            self.images[name]['clusters'][key]['outline'] = outline

    def outline_clusters_cv2(self, name='new_image'):  # Could be better, with inner outlines
        """Calculate with cv2 the outline of the clusters in the image."""
        data = self.images[name]["binary_data"].copy()
        data[0, :] = 0
        data[-1, :] = 0
        data[:, 0] = 0
        data[:, -1] = 0

        # RETR_EXTERNAL retrieve only the parent outlines and not the children, use RETR_TREE
        contours = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        data = [contours[i].flatten() for i in range(len(contours))]
        data = [data[i].reshape(int(len(data[i])/2), 2) for i in range(len(data))]

        self.images[name]['outlines'] = data

    def correct_outlines_cv2(self, name='new_image', actual_min=1, target_min=0.1,
                             actual_max=98, target_max=99.9):
        """Correct the outlines.

        Parameters
        ----------
        name : TYPE, optional
            DESCRIPTION. The default is 'new_image'.
        actual_min : TYPE, optional
            DESCRIPTION. The default is 1.
        target_min : TYPE, optional
            DESCRIPTION. The default is 0.1.
        actual_max : TYPE, optional
            DESCRIPTION. The default is 98.
        target_max : TYPE, optional
            DESCRIPTION. The default is 99.9.

        Returns
        -------
        None.
        """
        deduction = actual_min - target_min
        ratio = (target_max - target_min) / (actual_max - actual_min)
        data = self.images[name]['outlines'].copy()
        for i in range(len(data)):
            data[i] = data[i].astype(float)
            data[i][:, 1] = data[i][:, 1]-deduction
            data[i][:, 1] = data[i][:, 1]*ratio
        self.images[name]['outlines'] = data

# ==== SHOW FUNCTIONS ====

    def show_image(self, name='new_image'):
        """Display the greyscale image.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
        ax0.imshow(self.images[name]['data'], cmap='gray')
        ax0.set_xlabel('pixels')
        ax0.set_ylabel('pixels')
        plt.title(name)
        plt.show()

    def show_threshold_image(self, name='new_image'):
        """Display the image in two color in function of the threshold.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
        ax0.imshow(self.images[name]['binary_data'], cmap='gray')
        ax0.set_xlabel('pixels / um')
        ax0.set_ylabel('pixels / um')
        plt.title(name)
        plt.show()

    def show_t_i_with_i(self, name='new_image'):
        """Display the image in greyscale next to the binarized image.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        data = self.images[name]['data'].copy()
        data[data <= self.images[name]['threshold']] = False
        data[data > self.images[name]['threshold']] = True
        self.images[name]['binary_data'] = data
        fig, axes = plt.subplots(1, 2, figsize=[12, 5])
        axes[0].imshow(self.images[name]['data'], cmap='gray')
        axes[1].imshow(data, cmap='gray')
        axes[0].set_xlabel('pixels')
        axes[0].set_ylabel('pixels')
        axes[1].set_xlabel('pixels / um')
        axes[1].set_ylabel('pixels / um')
        plt.title(name)
        plt.show()

    def show_histogram(self, name='new_image', show_threshold=True):
        """Display the histogram of the image's pixels color.

        Parameters
        ----------
        name : string
            The name given to the image.
        nb_pools : int
            The number of pools for the histogram.
        show_threshold : bool
            If True show the line of the threshold.

        Returns
        -------
        None.
        """
        fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
        ax0.hist(self.images[name]['data'].flatten(),
                 bins=256, range=(-0.5, 255.5), density=True, facecolor='midnightblue', alpha=0.8)
        if show_threshold:
            ax0.axvline(self.images[name]['threshold'],
                        color='k', linestyle='dashed', linewidth=2)
        ax0.set_xlabel('pixel_values')
        ax0.set_ylabel('percentage')
        ax0.grid()
        plt.title(name)
        plt.show()

    def show_fft_2d_image(self, name='new_image'):
        """Display the 2D fft of the image.

        Parameters
        ----------
        name : string
            The name given to the image.

        Returns
        -------
        None.
        """
        fft_2d = fourier.fftn(self.images[name]['data'])
        fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
        ax0.imshow(np.log(np.abs(fourier.fftshift(fft_2d))**2), cmap='gray')
        ax0.set_xlabel('pixels')
        ax0.set_ylabel('pixels')
        plt.title(name)
        plt.show()

    def show_clusters(self, name='new_image', cluster_name=None):
        """Display the clusters present in the image.

        Parameters
        ----------
        name : string
            The name given to the image.
        cluster_name : string
            If specified, display only the desired cluster.

        Returns
        -------
        None.
        """
        if cluster_name:
            fig, ax0 = plt.subplots(1, 1, figsize=[12, 8])
            min_height = self.images[name]['clusters'][cluster_name]['min_row']
            max_height = self.images[name]['clusters'][cluster_name]['max_row']
            min_width = self.images[name]['clusters'][cluster_name]['min_column']
            max_width = self.images[name]['clusters'][cluster_name]['max_column']
            data = np.zeros((self.images[name]['height'], self.images[name]['width']), dtype=bool)
            data[self.images[name]['clusters'][cluster_name]['rows'],
                 self.images[name]['clusters'][cluster_name]['columns']] = True
            ax0.imshow(data[min_height:max_height+1, min_width:max_width+1], cmap='gray')
            ax0.set_xlabel(cluster_name + ' : pixels / um')
            ax0.set_ylabel('pixels / um')
        else:
            nb_clusters = len(self.images[name]['clusters'])
            fig, axes = plt.subplots(1, nb_clusters, figsize=[12, 8])
            for counter, key in enumerate(self.images[name]['clusters']):
                min_height = self.images[name]['clusters'][key]['min_row']
                max_height = self.images[name]['clusters'][key]['max_row']
                min_width = self.images[name]['clusters'][key]['min_column']
                max_width = self.images[name]['clusters'][key]['max_column']
                data = np.zeros((self.images[name]['height'], self.images[name]['width']), dtype=bool)
                data[self.images[name]['clusters'][key]['rows'], self.images[name]['clusters'][key]['columns']] = True
                axes[counter].imshow(data[min_height:max_height+1, min_width:max_width+1], cmap='gray')
                axes[counter].set_xlabel(key + ' : pixels / um')
                axes[counter].set_ylabel('pixels / um')
        plt.title(name)
        plt.show()

    def show_cluster_3D(self, name='new_image', cluster_name='c1', av_thick=10):
        """Display the cluster in 3D.

        Parameters
        ----------
        name : string
            The name given to the image.
        cluster_name : string
            If specified, display only the desired cluster.
        av_thick : int
            The average thickness of the cluters in terms of number of particles.

        Returns
        -------
        None.
        """
        min_height = self.images[name]['clusters'][cluster_name]['min_row']
        max_height = self.images[name]['clusters'][cluster_name]['max_row']
        min_width = self.images[name]['clusters'][cluster_name]['min_column']
        max_width = self.images[name]['clusters'][cluster_name]['max_column']
        data = np.zeros((self.images[name]['height'], self.images[name]['width']), dtype=bool)
        data[self.images[name]['clusters'][cluster_name]['rows'],
             self.images[name]['clusters'][cluster_name]['columns']] = True
        data = data[min_height:max_height+1, min_width:max_width+1]
        xs, ys = data.nonzero()
        zs = np.array(list(range(av_thick))*len(xs))
        xs = np.array(xs.tolist()*av_thick)
        ys = np.array(ys.tolist()*av_thick)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, c="k", s=200, alpha=0.5)
        # ax.plot(xs, ys, zs, c="k", alpha=0.5)
        # max_range = np.amax([np.amax(xs), np.amax(ys), np.amax(zs)])
        # ax.set_xlim3d(0, max_range)
        # ax.set_ylim3d(0, max_range)
        # ax.set_zlim3d(0, max_range)
        plt.show()

# ==== FEMM FUNCTIONS ====

    def femm2d(self, circles_parameters=None, clusters_parameters=None,
               w=1000, e=100, t=20, close=False, return_B=True, file_name="test.FEM"):
        """Do the calculation with FEMM 2D.

        Parameters
        ----------
        circles_parameters : TYPE, optional
            DESCRIPTION. The default is None.
        clusters_parameters : TYPE, optional
            DESCRIPTION. The default is None.
        w : TYPE, optional
            DESCRIPTION. The default is 1000.
        e : TYPE, optional
            DESCRIPTION. The default is 100.
        t : TYPE, optional
            DESCRIPTION. The default is 20.
        close : TYPE, optional
            DESCRIPTION. The default is False.
        return_B : TYPE, optional
            DESCRIPTION. The default is True.
        file_name : TYPE, optional
            DESCRIPTION. The default is "test.FEM".

        Returns
        -------
        permeance : TYPE
            DESCRIPTION.
        """
        femm.openfemm()
        femm.newdocument(0)
        femm.mi_probdef(0, 'micrometers', 'planar', 10**(-8), 1, 30)

        h = w / 10   # thickness of coil

        # Drawing the iron core
        femm.mi_drawline(0, 0, w, 0)
        femm.mi_drawline(w, 0, w, -w)
        femm.mi_drawline(w, -w, 2*w, -w)
        femm.mi_drawline(2*w, -w, 2*w, w+e)
        femm.mi_drawline(2*w, w+e, w, w+e)
        femm.mi_drawline(w, w+e, w, e)
        femm.mi_drawline(w, e, 0, e)
        femm.mi_drawline(0, e, 0, 2*w+e)
        femm.mi_drawline(0, 2*w+e, 3*w, 2*w+e)
        femm.mi_drawline(3*w, 2*w+e, 3*w, -2*w)
        femm.mi_drawline(3*w, -2*w, 0, -2*w)
        femm.mi_drawline(0, -2*w, 0, 0)

        # drawing the coil rectangles
        femm.mi_drawrectangle(2*w-h, (e+w)/2, 2*w, (e-w)/2)
        femm.mi_drawrectangle(3*w, (e+w)/2, 3*w+h, (e-w)/2)

        # draw the circles
        if circles_parameters is not None:
            r = circles_parameters["r"]
            if "X" in circles_parameters and "Y" in circles_parameters:
                X = circles_parameters["X"]
                Y = circles_parameters["Y"]
            else:
                X, Y = self.get_circles_distribution(w, e, r, epsilon=circles_parameters["epsilon"],
                                                     nb_circles=circles_parameters["nb_circles"])
            for i in range(len(X)):
                femm.mi_drawarc(X[i]-r, Y[i], X[i]+r, Y[i], 180, 8)
                femm.mi_drawarc(X[i]+r, Y[i], X[i]-r, Y[i], 180, 8)

        # draw the clusters of MRF
        if clusters_parameters is not None:
            outlines = clusters_parameters["outlines"]
            for cluster in outlines:
                for point in range(len(cluster)-1):
                    femm.mi_drawline(cluster[point][0], cluster[point][1], cluster[point+1][0], cluster[point+1][1])
                femm.mi_drawline(cluster[-1][0], cluster[-1][1], cluster[0][0], cluster[0][1])

        # default boundary conditions
        femm.mi_makeABC()

        # materials
        femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        femm.mi_addmaterial('Copper', 1, 1, 0, 0, 100, 0, 0, 1, 0, 0, 0)
        femm.mi_addmaterial('LinearCarbonylIron', 2100, 2100, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        femm.mi_addmaterial('LinearMRF', 10, 10, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        femm.mi_addmaterial('LinearPerfectIron', 10000, 10000, 0, 0, 0, 0, 0, 1, 0, 0, 0)

        # circuit
        femm.mi_addcircprop('Coil', 0.1, 1)

        # blocks labels
        femm.mi_addblocklabel(1.5*w, e/2)
        femm.mi_selectlabel(1.5*w, e/2)
        femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)
        femm.mi_clearselected()

        femm.mi_addblocklabel(2.5*w, e/2)
        femm.mi_selectlabel(2.5*w, e/2)
        femm.mi_setblockprop('LinearPerfectIron', 1, 0, '<None>', 0, 0, 0)
        femm.mi_clearselected()

        femm.mi_addblocklabel(2*w-h/2, e/2)
        femm.mi_selectlabel(2*w-h/2, e/2)
        femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, 1)
        femm.mi_clearselected()

        femm.mi_addblocklabel(3*w+h/2, e/2)
        femm.mi_selectlabel(3*w+h/2, e/2)
        femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, 1)
        femm.mi_clearselected()

        if circles_parameters is not None:
            for i in range(len(X)):
                femm.mi_addblocklabel(X[i], Y[i])
                femm.mi_selectlabel(X[i], Y[i])
                femm.mi_setblockprop('LinearCarbonylIron', 1, 0, '<None>', 0, 0, 0)
                femm.mi_clearselected()

        if clusters_parameters is not None:
            femm.mi_drawarc(1.5*w-h, 3*w, 1.5*w+h, 3*w, 180, 8)
            femm.mi_drawarc(1.5*w+h, 3*w, 1.5*w-h, 3*w, 180, 8)
            femm.mi_addblocklabel(1.5*w, 3*w)
            femm.mi_selectlabel(1.5*w, 3*w)
            femm.mi_setblockprop('LinearMRF', 1, 0, '<None>', 0, 0, 0)
            femm.mi_attachdefault()
            femm.mi_clearselected()

        # display correctly
        femm.mi_zoomnatural()

        # save it
        femm.mi_saveas(file_name)

        # run the calculations
        femm.mi_analyze()
        femm.mi_loadsolution()

        B_x = [abs(femm.mo_getb(i, e/2)[1]) for i in range(int(w))]
        H_y = [abs(femm.mo_geth(w/2, i)[1]) for i in range(int(-w), int(w+e))]
        delta_potential = np.sum(H_y)*10**(-6)
        section = w * t * 10**(-12)
        permeance = np.mean(B_x)*section / delta_potential

        if return_B:
            By = np.array([abs(femm.mo_getb(int(i/e), i % e)[1]) for i in range(int(w*e))])
            By = By.reshape(w, e)
            By = By.transpose()

        # Close FEMM 2D
        if close:
            femm.closefemm()

        if return_B:
            return (permeance, By)
        else:
            return permeance

# ==== APPLY FUNCTIONS ====

    def apply_to_all_im(self, functions_set=None):
        """Apply a set of function to all images stocked.

        Parameters
        ----------
        functions_set : list of strings
            The list of operations that we want to be applied, the names are the names of the related functions.

        Returns
        -------
        None.
        """
        if not functions_set:
            functions_set = self.default_parameters["functions_set"]

        for function in functions_set:
            for key in self.images:
                if function == "crop":
                    self.crop(name=key,
                              min_height=self.default_parameters["min_height"],
                              max_height=self.default_parameters["max_height"],
                              min_width=self.default_parameters["min_width"],
                              max_width=self.default_parameters["max_width"])
                if function == "image_size_change":
                    self.image_size_change(name=key,
                                           div_ratio=self.default_parameters["div_ratio"])
                if function == "convert_grey":
                    self.convert_grey(name=key)
                if function == "reverse_color":
                    self.reverse_color(name=key)
                if function == "color_threshold":
                    self.color_threshold(name=key,
                                         color_threshold=self.default_parameters["color_threshold"])
                if function == "show_histogram":
                    self.show_histogram(name=key,
                                        show_threshold=self.default_parameters["show_threshold"])
                if function == "show_image":
                    self.show_image(name=key)
                if function == "show_threshold_image":
                    self.show_threshold_image(name=key)
                if function == "show_t_i_with_i":
                    self.show_t_i_with_i(name=key)
                if function == "show_fft_2d_image":
                    self.show_fft_2d_image(name=key)
                if function == "show_show_clusters":
                    self.show_clusters(name=key)
                if function == "cluster_identification":
                    self.cluster_identification(name=key,
                                                tol_cluster=self.default_parameters["tol_cluster"])
                if function == "calculation_clusters_permeance":
                    self.calculation_clusters_permeance(name=key,
                                                        av_thick=self.default_parameters["av_thick"],
                                                        particle_permeance=self.default_parameters["particle_permeance"],
                                                        oil_permeance=self.default_parameters["oil_permeance"])
                if function == "outline_clusters_cv2":
                    self.outline_clusters_cv2(name=key)
                if function == "corr_with_curve":
                    self.corr_with_curve(name=key)
                if function == "correct_outlines_cv2":
                    self.correct_outlines_cv2(name=key,
                                              actual_min=self.default_parameters["actual_min"],
                                              target_min=self.default_parameters["target_min"],
                                              actual_max=self.default_parameters["actual_max"],
                                              target_max=self.default_parameters["target_max"])
