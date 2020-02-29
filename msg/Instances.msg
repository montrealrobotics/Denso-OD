# A list of 2D detections, for a multi-object 2D detector.

Header header

# A list of the detected proposals. A multi-proposal detector might generate
#   this list with many candidate detections generated from a single input.

BoundingBox2D[] detections
Variance2D[] variances
Classification2D[] classifications

float64 width
float64 height
string image_path
