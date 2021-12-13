# Stations Dataset

Stations Dataset was found at: https://cpcb.nic.in/nwmp-monitoring-network/

This folder contains code to visually represent the water quality predictions on the map of India using 5px x 5px squares with color coding:

- very good: green
- good: blue
- bad: orange
- very bad: red

There is a small offset due to the curves on the planet, which can be fixed, but in this case is not.

![visualisation](Resulting_Images/segmentedImage_original_dataset1.jpg "visualization")

To be able to do this, we first added the longitude and latitude to each station in the predicted dataset  using code_stations_processing notebook.

#NOTE: Min and max longitude nad latitude are gotten from openstreetmaps.

Once that is done, we can feed the output/dataset with coordinates into the full_segmentation notebook that will result in an image.
