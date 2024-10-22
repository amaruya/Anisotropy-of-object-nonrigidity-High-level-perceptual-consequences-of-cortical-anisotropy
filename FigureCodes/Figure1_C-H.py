#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Here, you can generate videos in Fig 1. You can select a shape from Circ (Circular rings) and Oct (Octagon rings)
as well as a type of motion from Rot (rotation) or Wob (wobbling)



"""

import Make_rotating_two_rings



# Make Horizontally rotating circular rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Circ',Motion_type='Rot',im_rot=0)
video_maker.forward()


# Make Vertically rotating circular rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Circ',Motion_type='Rot',im_rot=90)
video_maker.forward()

# Make Horizontally rotating octagon rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Oct',Motion_type='Rot',im_rot=0)
video_maker.forward()


# Make Vertically rotating Octagon rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Oct',Motion_type='Rot',im_rot=90)
video_maker.forward()



# Wobbling Stim

# Make Horizontally wobbling circular rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Circ',Motion_type='Wob',im_rot=0)
video_maker.forward()


# Make Vertically wobbling circular rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Circ',Motion_type='Wob',im_rot=90)
video_maker.forward()

# Make Horizontally wobbling octagon rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Oct',Motion_type='Wob',im_rot=0)
video_maker.forward()


# Make Vertically wobbling Octagon rings

video_maker=Make_rotating_two_rings.TwoRingsStim(Type='Oct',Motion_type='Wob',im_rot=90)
video_maker.forward()





