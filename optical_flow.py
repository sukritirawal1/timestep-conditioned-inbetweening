import cv2 
import numpy as np 
import torch 

class OpticalFlow:
    """
    Simple Optical Flow baseline for Frame Interpolation.
    Uses cv2.calcOpticalFlowFarneback to compute the optical flow between two frames.
    References: 
        - https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        - https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af 
        - Farneback, G. (2003). "Two-Frame Motion Estimation Based on Polynomial Expansion"
    """

    def __init__(self,pyr_scale=0.5, levels=3, winsize= 15, iterations=3, poly_n=5, poly_sigma=1.2 ):
        """ 
        Initialize optical flow parameters.

        Args:
            pyr_scale: pyramid scale factor (0.5 = each layer is half size)
            levels: number of pyramid levels
            winsize: averaging window size
            iterations: iterations at each pyramid level
            poly_n: size of pixel neighborhood (5 or 7)
            poly_sigma: standard deviation for Gaussian smoothing
        """

        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
    
    def interpolate(self, start_frame, end_frame, t):
        """
        Generate intermediate frame at timestep t between start_frame and end_frame.

        Args:
            start_frame: first frame
            end_frame: last frame
            t: = float in [0,1], interpolation timpstep (0.25, 0.5, 0.75)

        """
        # convert to grayscale
        # [C,H,W] -> [H,W,C]
        start_np = (start_frame.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        end_np = (end_frame.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        
        start_gray = cv2.cvtColor(start_np, cv2.COLOR_RGB2GRAY)
        end_gray = cv2.cvtColor(end_np, cv2.COLOR_RGB2GRAY)
        
        # compute the bidirectional flow
        flow_forward = self._compute_flow(start_gray, end_gray)
        flow_backward = self._compute_flow(end_gray, start_gray)

        # warp both frames towards timestep t 
        warped_start = self._warp_frame(start_np, flow_forward, t)
        warped_end = self._warp_frame(end_np, flow_backward, 1-t)

        # belend the warped frames 
        blended_frame = ((1-t) * warped_start + t * warped_end).astype(np.uint8)

        #convert back to torch tensor [C,H,W], float32
        result = torch.from_numpy(blended_frame).permute(2,0,1).float() / 255.0

        return result 
        #pass

    def _compute_flow(self, gray1, gray2):
        """
        Compute the dense optical flow from gray1 to gray2.
        Args: 
            gray1, gray2: numpy arrays, uint8, single channel
        Returns:
            flow: numpy array [H,W,2], motion vectors (dx, dy)

        """
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            self.pyr_scale, 
            self.levels, 
            self.winsize, 
            self.iterations, 
            self.poly_n, 
            self.poly_sigma,
            flags = 0 
        )
        return flow 

    def _warp_frame(self, frame, flow, t):
        """
        Warp frame using the optical flow scaled by timestep t.
        """
        h,w = frame.shape[:2]

        flow_scaled = flow * t
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + flow_scaled[..., 0]).astype(np.float32)
        map_y = (y + flow_scaled[..., 1]).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return warped



            




