#include "texture.h"
#include "color.h"

#include <assert.h>
#include <iostream>
#include <algorithm>

using namespace std;

namespace CS248 {

inline void uint8_to_float( float dst[4], unsigned char* src ) {
  uint8_t* src_uint8 = (uint8_t *)src;
  dst[0] = src_uint8[0] / 255.f;
  dst[1] = src_uint8[1] / 255.f;
  dst[2] = src_uint8[2] / 255.f;
  dst[3] = src_uint8[3] / 255.f;
}

inline void float_to_uint8( unsigned char* dst, float src[4] ) {
  uint8_t* dst_uint8 = (uint8_t *)dst;
  dst_uint8[0] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[0])));
  dst_uint8[1] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[1])));
  dst_uint8[2] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[2])));
  dst_uint8[3] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[3])));
}

void Sampler2DImp::generate_mips(Texture& tex, int startLevel) {

  // NOTE: 
  // This starter code allocates the mip levels and generates a level 
  // map by filling each level with placeholder data in the form of a 
  // color that differs from its neighbours'. You should instead fill
  // with the correct data!

  // Advanced Task
  // Implement mipmap for trilinear filtering

  // check start level
  if ( startLevel >= tex.mipmap.size() ) {
    std::cerr << "Invalid start level"; 
  }

  // allocate sublevels
  int baseWidth  = tex.mipmap[startLevel].width;
  int baseHeight = tex.mipmap[startLevel].height;
  int numSubLevels = (int)(log2f( (float)max(baseWidth, baseHeight)));

  numSubLevels = min(numSubLevels, kMaxMipLevels - startLevel - 1);
  tex.mipmap.resize(startLevel + numSubLevels + 1);

  int width  = baseWidth;
  int height = baseHeight;
  for (int i = 1; i <= numSubLevels; i++) {

    MipLevel& level = tex.mipmap[startLevel + i];

    // handle odd size texture by rounding down
    width  = max( 1, width  / 2); assert(width  > 0);
    height = max( 1, height / 2); assert(height > 0);

    level.width = width;
    level.height = height;
    level.texels = vector<unsigned char>(4 * width * height);

  }

  // fill all 0 sub levels with interchanging colors
  // Color colors[3] = { Color(1,0,0,1), Color(0,1,0,1), Color(0,0,1,1) };
  // for(size_t i = 1; i < tex.mipmap.size(); ++i) {

  //   Color c = colors[i % 3];
  //   MipLevel& mip = tex.mipmap[i];

  //   for(size_t i = 0; i < 4 * mip.width * mip.height; i += 4) {
  //     float_to_uint8( &mip.texels[i], &c.r );
  //   }
  // }

  for (int level = startLevel + 1; level < tex.mipmap.size(); level++) {
    MipLevel& prevLevel = tex.mipmap[level - 1];
    MipLevel& currLevel = tex.mipmap[level];

    for (int y = 0; y < currLevel.height; y++) {
      for (int x = 0; x < currLevel.width; x++) {
        int baseX = x * 2;
        int baseY = y * 2;

        Color avgColor = {0, 0, 0, 0};
        for (int dy = 0; dy < 2; dy++) {
          for (int dx = 0; dx < 2; dx++) {
            int clampedX = std::min(baseX + dx, static_cast<int>(prevLevel.width) - 1);
            int clampedY = std::min(baseY + dy, static_cast<int>(prevLevel.height) - 1);
            int srcIdx = 4 * (clampedY * prevLevel.width + clampedX);
            avgColor.r += prevLevel.texels[srcIdx];
            avgColor.g += prevLevel.texels[srcIdx + 1];
            avgColor.b += prevLevel.texels[srcIdx + 2];
            avgColor.a += prevLevel.texels[srcIdx + 3];
          }
        }

        // Average the color
        avgColor.r /= 4;
        avgColor.g /= 4;
        avgColor.b /= 4;
        avgColor.a /= 4;

        // Set the pixel in the current level
        int dstIdx = 4 * (y * currLevel.width + x);
        if (dstIdx + 3 < currLevel.texels.size()) {
          float_to_uint8(&currLevel.texels[dstIdx], &avgColor.r);
        }
      }
    }
  }
}

Color Sampler2DImp::sample_nearest(Texture& tex, 
                                   float u, float v, 
                                   int level) {

  // Task 4: Implement nearest neighbour interpolation
  if (level != 0) { // return magenta for invalid level
    return Color(1,0,1,1);
  }

  MipLevel &mip = tex.mipmap[level];
  int texelX = static_cast<int>(u * mip.width);
  int texelY = static_cast<int>(v * mip.height);
  int texelIndex = 4 * (texelY * mip.width + texelX);

  return Color(mip.texels[texelIndex] / 255.f,
               mip.texels[texelIndex + 1] / 255.f,
               mip.texels[texelIndex + 2] / 255.f,
               mip.texels[texelIndex + 3] / 255.f);
}

Color Sampler2DImp::sample_bilinear(Texture& tex, 
                                    float u, float v, 
                                    int level) {
  
  // Task 4: Implement bilinear filtering
  if (level != 0) { // return magenta for invalid level
    return Color(1,0,1,1);
  }

  MipLevel &mip = tex.mipmap[level];
  float texelX = u * mip.width;
  float texelY = v * mip.height;

  int x0 = std::max(0, static_cast<int>(texelX - 0.5f));
  int y0 = std::max(0, static_cast<int>(texelY - 0.5f));
  int x1 = std::min(x0 + 1, static_cast<int>(mip.width) - 1);
  int y1 = std::min(y0 + 1, static_cast<int>(mip.height) - 1);

  auto get_color = [&](MipLevel &mip, int x, int y) {
    int texelIndex = 4 * (y * mip.width + x);
    return Color(mip.texels[texelIndex] / 255.f,
                 mip.texels[texelIndex + 1] / 255.f,
                 mip.texels[texelIndex + 2] / 255.f,
                 mip.texels[texelIndex + 3] / 255.f);
  };

  Color c00 = get_color(mip, x0, y0);
  Color c10 = get_color(mip, x1, y0);
  Color c01 = get_color(mip, x0, y1);
  Color c11 = get_color(mip, x1, y1);

  float u_ratio = texelX - (x0 + 0.5);
  float v_ratio = texelY - (y0 + 0.5);

  Color c0 = c00 * (1 - u_ratio) + c10 * u_ratio;
  Color c1 = c01 * (1 - u_ratio) + c11 * u_ratio;

  return c0 * (1 - v_ratio) + c1 * v_ratio;
}

Color Sampler2DImp::sample_trilinear(Texture& tex, 
                                     float u, float v, 
                                     float u_scale, float v_scale) {

  // Advanced Task
  // Implement trilinear filtering
  auto get_color = [&](MipLevel &mip, int x, int y) {
    int texelIndex = 4 * (y * mip.width + x);
    return Color(mip.texels[texelIndex] / 255.f,
                 mip.texels[texelIndex + 1] / 255.f,
                 mip.texels[texelIndex + 2] / 255.f,
                 mip.texels[texelIndex + 3] / 255.f);
  };

  auto clamp = [](int value, int minVal, int maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
  };

  float L = log2f(std::max(u_scale, v_scale));
  int level1 = static_cast<int>(L);
  int level2 = level1 + 1;
  level1 = clamp(level1, 0, static_cast<int>(tex.mipmap.size()) - 1);
  level2 = clamp(level2, 0, static_cast<int>(tex.mipmap.size()) - 1);

  MipLevel &mip1 = tex.mipmap[level1];
  float texelX1 = u * mip1.width;
  float texelY1 = v * mip1.height;
  int x0 = std::max(0, static_cast<int>(texelX1 - 0.5f));
  int y0 = std::max(0, static_cast<int>(texelY1 - 0.5f));
  int x1 = std::min(x0 + 1, static_cast<int>(mip1.width) - 1);
  int y1 = std::min(y0 + 1, static_cast<int>(mip1.height) - 1);
  Color c00 = get_color(mip1, x0, y0);
  Color c10 = get_color(mip1, x1, y0);
  Color c01 = get_color(mip1, x0, y1);
  Color c11 = get_color(mip1, x1, y1);
  float u_ratio = texelX1 - (x0 + 0.5);
  float v_ratio = texelY1 - (y0 + 0.5);
  Color c0 = c00 * (1 - u_ratio) + c10 * u_ratio;
  Color c1 = c01 * (1 - u_ratio) + c11 * u_ratio;
  Color c = c0 * (1 - v_ratio) + c1 * v_ratio;

  MipLevel &mip2 = tex.mipmap[level2];
  float texelX2 = u * mip2.width;
  float texelY2 = v * mip2.height;
  x0 = std::max(0, static_cast<int>(texelX2 - 0.5f));
  y0 = std::max(0, static_cast<int>(texelY2 - 0.5f));
  x1 = std::min(x0 + 1, static_cast<int>(mip2.width) - 1);
  y1 = std::min(y0 + 1, static_cast<int>(mip2.height) - 1);
  c00 = get_color(mip2, x0, y0);
  c10 = get_color(mip2, x1, y0);
  c01 = get_color(mip2, x0, y1);
  c11 = get_color(mip2, x1, y1);
  u_ratio = texelX2 - (x0 + 0.5);
  v_ratio = texelY2 - (y0 + 0.5);
  c0 = c00 * (1 - u_ratio) + c10 * u_ratio;
  c1 = c01 * (1 - u_ratio) + c11 * u_ratio;
  Color c2 = c0 * (1 - v_ratio) + c1 * v_ratio;

  float t = L - level1;
  return c * (1 - t) + c2 * t;
}

} // namespace CS248
