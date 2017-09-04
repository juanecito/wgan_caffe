#!/bin/bash
################################################################################
# 
# * create_gif_animation.sh
# * Copyright (C) 2017 Juan Maria Gomez Lopez <juanecitorr@gmail.com>
# *
# * caffe_network is free software: you can redistribute it and/or modify it
# * under the terms of the GNU General Public License as published by the
# * Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
# *
# * caffe_wgan is distributed in the hope that it will be useful, but
# * WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# * See the GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License along
# * with this program.  If not, see <http://www.gnu.org/licenses/>.
# */

#* @file create_gif_animation.sh
# * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
# * @date 20 Jun 2017
# */

################################################################################

#convert -delay 40 wgan_grid*.jpg -loop 0 cifar10.gif
convert -delay 20 wgan_grid*.jpg -loop 1 -comment "WGAN Caffe, Faces, Juan Maria GOMEZ LOPEZ" faces.gif

exit 0

