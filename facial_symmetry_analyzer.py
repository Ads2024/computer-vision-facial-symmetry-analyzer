import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import io

class FacialSymmetryAnalyzer:
    """
    A comprehensive facial symmetry analyzer that uses MediaPipe Face Mesh to detect landmarks
    and calculate symmetry metrics across different facial zones.
    """
    
    def __init__(self):
        # MediaPipe face mesh setup with higher confidence threshold for better accuracy
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Define facial symmetry zones with their corresponding landmark pairs
        self.symmetry_zones = {
            "eyebrows": [(65, 295), (66, 296), (107, 336), (105, 334), (70, 300), (46, 276)],
            "eyes": [(33, 263), (133, 362), (160, 387), (159, 386), (158, 385), (157, 384)],
            "nose": [(168, 388), (2, 98), (4, 327), (19, 242), (126, 352)],
            "lips": [(61, 291), (57, 287), (62, 292), (76, 306), (77, 307), (90, 320)],
            "jaw": [(199, 429), (198, 428), (197, 427), (196, 426), (177, 401), (148, 377)],
            "cheeks": [(234, 454), (93, 323), (132, 361), (127, 356), (138, 365)],
        }
        
        # Perceptual importance weights
        self.perceptual_weights = {
            "eyebrows": 0.7,
            "eyes": 1.0,
            "nose": 0.8,
            "lips": 0.9,
            "jaw": 0.6,
            "cheeks": 0.5,
        }
    
    def analyze_image(self, image_path, use_perceptual_weights=True, visualize=True):
        """
        Analyze facial symmetry in an image.
        """
        try:
            # Load and check image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image at {image_path}")
            
            if img.size == 0:
                raise ValueError("Image is empty")
            
            print(f"Image loaded successfully. Shape: {img.shape}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            # Process with MediaPipe
            results = self.face_mesh.process(img_rgb)
            
            if not results.multi_face_landmarks:
                raise ValueError("No face detected in the image")
            
            print("Face detected successfully")
            
            # Extract landmark coordinates
            landmarks = results.multi_face_landmarks[0].landmark
            points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
            
            # Calculate face dimensions for normalization
            face_height = abs(np.mean([points[10][1], points[152][1]]) - 
                            np.mean([points[8][1], points[168][1]]))
            face_width = abs(points[234][0] - points[454][0])
            
            if face_height <= 0 or face_width <= 0:
                raise ValueError("Invalid face dimensions detected")
            
            print(f"Face dimensions calculated. Height: {face_height}, Width: {face_width}")
            
            # Align face upright based on eye positions
            aligned_img, aligned_points, mid_x = self._align_face(img, points)
            print("Face aligned successfully")
            
            # Calculate symmetry scores
            zone_scores, all_landmarks_scores, asymmetry_map = self._calculate_symmetry(
                aligned_points, mid_x, face_width, face_height, h, w
            )
            print("Symmetry scores calculated")
            
            # Calculate final symmetry score
            if use_perceptual_weights:
                final_score = sum(zone_scores[zone] * self.perceptual_weights[zone] 
                                for zone in zone_scores) / sum(self.perceptual_weights.values())
            else:
                final_score = np.mean([score for score in zone_scores.values()])
            
            print(f"Final symmetry score: {final_score}")
            
            # Sort landmarks by asymmetry
            sorted_asym = sorted(all_landmarks_scores, key=lambda x: x[3], reverse=True)
            
            # Create visualization
            fig = self._create_visualization(aligned_img, aligned_points, mid_x, asymmetry_map, 
                                           zone_scores, final_score, use_perceptual_weights)
            print("Visualization created")
            
            # Save visualization to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            plt.close(fig)
            
            # Create mirrored versions
            left_half = aligned_img[:, :mid_x]
            right_half = aligned_img[:, mid_x:]
            mirrored_left = cv2.flip(left_half, 1)
            mirrored_right = cv2.flip(right_half, 1)
            
            # Ensure mirrored halves have the same width
            if mirrored_left.shape[1] != right_half.shape[1]:
                mirrored_left = cv2.resize(mirrored_left, (right_half.shape[1], right_half.shape[0]))
            if mirrored_right.shape[1] != left_half.shape[1]:
                mirrored_right = cv2.resize(mirrored_right, (left_half.shape[1], left_half.shape[0]))
            
            # Create mirrored images
            left_mirror = np.zeros_like(aligned_img)
            right_mirror = np.zeros_like(aligned_img)
            left_mirror[:, :mid_x] = aligned_img[:, :mid_x]
            left_mirror[:, mid_x:] = mirrored_left
            right_mirror[:, :mid_x] = mirrored_right
            right_mirror[:, mid_x:] = aligned_img[:, mid_x:]
            
            print("Mirrored images created")
            
            # Prepare results
            results = {
                "total_score": final_score,
                "zone_scores": zone_scores,
                "top_asymmetrical": sorted_asym[:5],
                "visualization": cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR),
                "left_mirror": left_mirror,
                "right_mirror": right_mirror
            }
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_image: {str(e)}")
            raise
    
    def _align_face(self, img, points):
        """Align face upright based on eye positions"""
        h, w, _ = img.shape
        
        # Find eye centers
        left_eye = np.mean([points[33], points[133]], axis=0)
        right_eye = np.mean([points[362], points[263]], axis=0)
        
        # Calculate rotation angle
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Use eye center as rotation center
        center_coords = np.mean([left_eye, right_eye], axis=0)
        center = (float(center_coords[0]), float(center_coords[1]))
        
        # Rotate image to align eyes horizontally
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_img = cv2.warpAffine(img, rot_matrix, (w, h))
        
        # Re-run face mesh on aligned image
        aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(aligned_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face found after alignment")
        
        # Extract aligned landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        aligned_points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        
        # Find vertical midline
        nose_tip = aligned_points[4]
        chin = aligned_points[152]
        mid_x = int((nose_tip[0] + chin[0]) / 2)
        
        return aligned_img, aligned_points, mid_x
    
    def _calculate_symmetry(self, aligned_points, mid_x, face_width, face_height, h, w):
        """Calculate symmetry scores for facial zones"""
        zone_scores = {}
        all_landmarks_scores = []
        asymmetry_map = np.zeros((h, w), dtype=np.float32)
        
        for zone_name, pairs in self.symmetry_zones.items():
            zone_score = []
            
            for left_idx, right_idx in pairs:
                try:
                    lx, ly = aligned_points[left_idx]
                    rx, ry = aligned_points[right_idx]
                    
                    # Mirror right side across calculated midline
                    mirrored_rx = 2 * mid_x - rx
                    
                    # Calculate Euclidean distance between mirrored right point and left point
                    euclidean_dist = np.sqrt((lx - mirrored_rx)**2 + (ly - ry)**2)
                    
                    # Normalize by face dimensions
                    norm_dist = euclidean_dist / np.sqrt(face_width * face_height)
                    zone_score.append(norm_dist)
                    
                    # Store landmark score for detailed analysis
                    all_landmarks_scores.append((zone_name, left_idx, right_idx, norm_dist))
                    
                    # Add to asymmetry map for visualization
                    asymmetry_map[ly, lx] = norm_dist
                    asymmetry_map[ry, rx] = norm_dist
                
                except IndexError:
                    print(f"Warning: Invalid landmark indices {left_idx}, {right_idx}")
            
            # Calculate mean score for this zone
            if zone_score:
                zone_scores[zone_name] = np.mean(zone_score)
        
        return zone_scores, all_landmarks_scores, asymmetry_map
    
    def _create_visualization(self, aligned_img, aligned_points, mid_x, asymmetry_map, 
                            zone_scores, total_score, use_perceptual_weights):
        """Create visualization of facial symmetry analysis"""
        h, w, _ = aligned_img.shape
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Face with landmarks and midline
        ax1.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        ax1.axvline(x=mid_x, color='r', linestyle='-', linewidth=1, alpha=0.7)
        
        # Plot landmarks with color-coded zones
        zone_colors = {
            "eyebrows": "blue",
            "eyes": "green",
            "nose": "purple",
            "lips": "red",
            "jaw": "orange",
            "cheeks": "cyan"
        }
        
        for zone, pairs in self.symmetry_zones.items():
            for left_idx, right_idx in pairs:
                lx, ly = aligned_points[left_idx]
                rx, ry = aligned_points[right_idx]
                ax1.scatter(lx, ly, color=zone_colors[zone], s=15, alpha=0.7)
                ax1.scatter(rx, ry, color=zone_colors[zone], s=15, alpha=0.7)
        
        ax1.set_title("Face Alignment & Landmarks")
        ax1.axis('off')
        
        # Plot 2: Asymmetry heatmap
        smoothed_map = gaussian_filter(asymmetry_map, sigma=5)
        masked_map = np.ma.masked_where(smoothed_map == 0, smoothed_map)
        
        ax2.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        heatmap = ax2.imshow(masked_map, cmap='hot', alpha=0.6, 
                           norm=Normalize(vmin=0, vmax=np.max(masked_map)*1.5))
        ax2.set_title("Asymmetry Heatmap")
        ax2.axis('off')
        
        plt.colorbar(heatmap, ax=ax2, shrink=0.7, label='Asymmetry Level')
        
        # Plot 3: Mirror comparison
        left_half = aligned_img[:, :mid_x]
        right_half = aligned_img[:, mid_x:]
        mirrored_left = cv2.flip(left_half, 1)
        
        if mirrored_left.shape[1] != right_half.shape[1]:
            mirrored_left = cv2.resize(mirrored_left, (right_half.shape[1], right_half.shape[0]))
        
        mirrored = np.zeros_like(aligned_img)
        mirrored[:, :mid_x] = aligned_img[:, :mid_x]
        mirrored[:, mid_x:] = mirrored_left
        
        ax3.imshow(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
        ax3.axvline(x=mid_x, color='r', linestyle='-', linewidth=1, alpha=0.7)
        ax3.set_title("Left Side Mirrored")
        ax3.axis('off')
        
        # Add overall title with score
        weight_text = " (perceptually weighted)" if use_perceptual_weights else ""
        plt.suptitle(f"Facial Symmetry Analysis - Score: {total_score:.4f}{weight_text}", 
                    fontsize=16, y=0.98)
        
        # Add zone scores as text
        score_text = "\n".join([f"{zone.capitalize()}: {score:.3f}" 
                               for zone, score in zone_scores.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax3.text(1.05, 0.5, score_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig 