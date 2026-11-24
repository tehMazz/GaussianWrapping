#ifndef CUDA_RASTERIZER_LABELS_H_INCLUDED
#define CUDA_RASTERIZER_LABELS_H_INCLUDED

// Optional BEA-GS label inputs/outputs. All pointers null = stock rasterizer.
//   votes[g * L + l] += alpha * T   for every pixel with label l, 0 <= l < L (label voting)
//   sag[p]           = sum alpha * T over Gaussians whose label != pixel label
// Pixels with label -1 (or >= L) are unlabeled: no votes, sag = 0.
// Optional GAVIS visibility (independent of labels), vis[g * 5 + c]:
//   c = 0: sum of T in front of g over the pixels g covers (alpha >= 1/255, before the T < 1e-4 stop)
//   c = 1: number of those pixels
//   c = 2: footprint area in pixels of g (alpha >= 1/255 ellipse), 0 when culled
//   c = 3: max of alpha * T over the LABELLED pixels (pixel_label >= 0) g covers; 0 without pixel labels.
//          "does g dominate some pixel", as opposed to c = 0..2 which only say how much it is occluded.
//   c = 4: sum of alpha * T over every pixel g covers (no label needed) - g's blend mass this render.
//          gs-atoms EpochPrune contribution signal; unlike c = 3 this is a sum, not a max, and is
//          written for unlabelled pixels too.
struct LabelArgs {
    const int* gaussian_labels = nullptr; // [P]
    const int* pixel_labels    = nullptr; // [H * W]
    int L                      = 0;       // number of labels
    float* sag                 = nullptr; // [H * W]
    float* votes               = nullptr; // [P * L]
    const float* dL_dsag       = nullptr; // [H * W], backward only
    float* vis                 = nullptr; // [P * 5], forward only
};

#endif
