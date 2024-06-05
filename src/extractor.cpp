#include "extractor.h"

struct greaterThanPtr
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) || *a >= *b && (a > b); }
};

/**
 * @brief
 * @param score_map
 * @param cell_size
 * @param kps
 * @return
 */
std::vector<cv::KeyPoint> nms(cv::InputArray score_map, int maxCorners,
                              double qualityLevel,double minDistance, cv::InputArray _mask)
{
    std::vector<cv::KeyPoint> kps;
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(score_map)));

    cv::Mat eig = score_map.getMat(), tmp;
    double maxVal = 0;
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
    cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO);
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size imgsize = eig.size();
    std::vector<const float*> tmpCorners;

    cv::Mat mask = _mask.getMat();
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const auto* eig_data = (const float*)eig.ptr(y);
        const auto* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<float> cornersQuality;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        return {};
    }

    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = eig.cols;
        int h = eig.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                cornersQuality.push_back(*tmpCorners[i]);
                kps.emplace_back((float)x, (float)y, 1.f);
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            cornersQuality.push_back(*tmpCorners[i]);

            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            kps.emplace_back((float)x, (float)y, 1.f);
            ++ncorners;

            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    return kps;
}

cv::Mat bilinear_interpolation(int image_w, int image_h, const cv::Mat& desc_map,
                               const std::vector<cv::KeyPoint>& kps) {
    int w = desc_map.cols;
    int h = desc_map.rows;
    int c = desc_map.channels();
    cv::Mat desc((int)kps.size(), c, CV_32F);
    for (int i = 0; i < kps.size(); i++) {
        cv::KeyPoint kp = kps[i];
        float x = kp.pt.x / (float)image_w * (float)w;
        float y = kp.pt.y / (float)image_h * (float)h;
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = std::min(x0 + 1, w - 1);
        int y1 = std::min(y0 + 1, h - 1);
        float dx = x - (float)x0;
        float dy = y - (float)y0;
        for (int j = 0; j < c; j++) {
            float v00 = desc_map.at<float>(y0, x0 * c + j);
            float v01 = desc_map.at<float>(y1, x0 * c + j);
            float v10 = desc_map.at<float>(y0, x1 * c + j);
            float v11 = desc_map.at<float>(y1, x1 * c + j);
            desc.at<float>(i, j) = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
                                   (1 - dx) * dy * v01 + dx * dy * v11;
        }
    }

    return desc;
}
