
uint32_t calNumEigenValuesLessThan(const float *diagonal,
                               const float *offDiagonal,
                               const uint32_t  length,
                               const float x);

uint32_t eigenValueCPUReference(float * diagonal,
                            float * offDiagonal,
                            uint32_t    length,
                            float * eigenIntervals,
                            float * newEigenIntervals,
                            float tolerance);

int isComplete(float * eigenIntervals, const int length, const float tolerance);

void computeGerschgorinInterval(float * lLimit,
                                float * uLimit,
                                const float * diagonal,
                                const float * offDiagonal,
                                const uint32_t  length);
