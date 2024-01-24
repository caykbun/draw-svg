#include "software_renderer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "triangulation.h"

using namespace std;

namespace CS248 {


// Implements SoftwareRenderer //

// fill a sample location with color
void SoftwareRendererImp::fill_sample(int sx, int sy, const Color &color) {
  // Task 2: implement this function
  int s_idx = sx + sy * width * sample_rate;

  Color sample_color;
	float inv255 = 1.0 / 255.0;
	sample_color.r = sample_buffer[4 * s_idx] * inv255;
	sample_color.g = sample_buffer[4 * s_idx + 1] * inv255;
	sample_color.b = sample_buffer[4 * s_idx + 2] * inv255;
	sample_color.a = sample_buffer[4 * s_idx + 3] * inv255;

  sample_color = alpha_blending(sample_color, color);

  sample_buffer[4 * s_idx] = (uint8_t)(sample_color.r * 255);
  sample_buffer[4 * s_idx + 1] = (uint8_t)(sample_color.g * 255);
  sample_buffer[4 * s_idx + 2] = (uint8_t)(sample_color.b * 255);
  sample_buffer[4 * s_idx + 3] = (uint8_t)(sample_color.a * 255);
}

// fill samples in the entire pixel specified by pixel coordinates
void SoftwareRendererImp::fill_pixel(int x, int y, const Color &color) {

	// Task 2: Re-implement this function

	// check bounds
	if (x < 0 || x >= width) return;
	if (y < 0 || y >= height) return;

  for (int xx = 0; xx < sample_rate; xx++) {
    for (int yy = 0; yy < sample_rate; yy++) {
      int sx = x * sample_rate + xx;
      int sy = y * sample_rate + yy;
      fill_sample(sx, sy, color);
    }
  }
}

void SoftwareRendererImp::draw_svg( SVG& svg ) {

  // set top level transformation
  transformation = canvas_to_screen;

  // canvas outline
  Vector2D a = transform(Vector2D(0, 0)); a.x--; a.y--;
  Vector2D b = transform(Vector2D(svg.width, 0)); b.x++; b.y--;
  Vector2D c = transform(Vector2D(0, svg.height)); c.x--; c.y++;
  Vector2D d = transform(Vector2D(svg.width, svg.height)); d.x++; d.y++;

  svg_bbox_top_left = Vector2D(a.x+1, a.y+1);
  svg_bbox_bottom_right = Vector2D(d.x-1, d.y-1);

  // draw all elements
  for (size_t i = 0; i < svg.elements.size(); ++i) {
    draw_element(svg.elements[i]);
  }

  // draw canvas outline
  rasterize_line(a.x, a.y, b.x, b.y, Color::Black);
  rasterize_line(a.x, a.y, c.x, c.y, Color::Black);
  rasterize_line(d.x, d.y, b.x, b.y, Color::Black);
  rasterize_line(d.x, d.y, c.x, c.y, Color::Black);

  // resolve and send to pixel buffer
  resolve();

}

void SoftwareRendererImp::set_sample_rate( size_t sample_rate ) {

  // Task 2: 
  // You may want to modify this for supersampling support
  this->sample_rate = sample_rate;
  resize_sample_buffer();
}

void SoftwareRendererImp::set_pixel_buffer( unsigned char* pixel_buffer,
                                             size_t width, size_t height ) {

  // Task 2: 
  // You may want to modify this for supersampling support
  this->pixel_buffer = pixel_buffer;
  this->width = width;
  this->height = height;
  resize_sample_buffer();
}

void SoftwareRendererImp::resize_sample_buffer() {
  // adjust the size for sample buffer to be at least number of pixels
	sample_buffer.resize(4 * width * height * sample_rate * sample_rate);
  clear_sample_buffer();
}

void SoftwareRendererImp::clear_sample_buffer() {
  memset(sample_buffer.data(), 255, 4 * width * height * sample_rate * sample_rate);
}

void SoftwareRendererImp::draw_element( SVGElement* element ) {

	// Task 3 (part 1):
	// Modify this to implement the transformation stack
  Matrix3x3 previous_transformation = transformation;
  transformation = transformation * element->transform;

	switch (element->type) {
	case POINT:
		draw_point(static_cast<Point&>(*element));
		break;
	case LINE:
		draw_line(static_cast<Line&>(*element));
		break;
	case POLYLINE:
		draw_polyline(static_cast<Polyline&>(*element));
		break;
	case RECT:
		draw_rect(static_cast<Rect&>(*element));
		break;
	case POLYGON:
		draw_polygon(static_cast<Polygon&>(*element));
		break;
	case ELLIPSE:
		draw_ellipse(static_cast<Ellipse&>(*element));
		break;
	case IMAGE:
		draw_image(static_cast<Image&>(*element));
		break;
	case GROUP:
		draw_group(static_cast<Group&>(*element));
		break;
	default:
		break;
	}

  transformation = previous_transformation;
}


// Primitive Drawing //

void SoftwareRendererImp::draw_point( Point& point ) {

  Vector2D p = transform(point.position);
  rasterize_point( p.x, p.y, point.style.fillColor );

}

void SoftwareRendererImp::draw_line( Line& line ) { 

  Vector2D p0 = transform(line.from);
  Vector2D p1 = transform(line.to);
  rasterize_line( p0.x, p0.y, p1.x, p1.y, line.style.strokeColor );

}

void SoftwareRendererImp::draw_polyline( Polyline& polyline ) {

  Color c = polyline.style.strokeColor;

  if( c.a != 0 ) {
    int nPoints = polyline.points.size();
    for( int i = 0; i < nPoints - 1; i++ ) {
      Vector2D p0 = transform(polyline.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polyline.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_rect( Rect& rect ) {

  Color c;
  
  // draw as two triangles
  float x = rect.position.x;
  float y = rect.position.y;
  float w = rect.dimension.x;
  float h = rect.dimension.y;

  Vector2D p0 = transform(Vector2D(   x   ,   y   ));
  Vector2D p1 = transform(Vector2D( x + w ,   y   ));
  Vector2D p2 = transform(Vector2D(   x   , y + h ));
  Vector2D p3 = transform(Vector2D( x + w , y + h ));
  
  // draw fill
  c = rect.style.fillColor;
  if (c.a != 0 ) {
    rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    rasterize_triangle( p2.x, p2.y, p1.x, p1.y, p3.x, p3.y, c );
  }

  // draw outline
  c = rect.style.strokeColor;
  if( c.a != 0 ) {
    rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    rasterize_line( p1.x, p1.y, p3.x, p3.y, c );
    rasterize_line( p3.x, p3.y, p2.x, p2.y, c );
    rasterize_line( p2.x, p2.y, p0.x, p0.y, c );
  }

}

void SoftwareRendererImp::draw_polygon( Polygon& polygon ) {

  Color c;

  // draw fill
  c = polygon.style.fillColor;
  if( c.a != 0 ) {

    // triangulate
    vector<Vector2D> triangles;
    triangulate( polygon, triangles );

    // draw as triangles
    for (size_t i = 0; i < triangles.size(); i += 3) {
      Vector2D p0 = transform(triangles[i + 0]);
      Vector2D p1 = transform(triangles[i + 1]);
      Vector2D p2 = transform(triangles[i + 2]);
      rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    }
  }

  // draw outline
  c = polygon.style.strokeColor;
  if( c.a != 0 ) {
    int nPoints = polygon.points.size();
    for( int i = 0; i < nPoints; i++ ) {
      Vector2D p0 = transform(polygon.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polygon.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_ellipse( Ellipse& ellipse ) {

  // Advanced Task
  // Implement ellipse rasterization

}

void SoftwareRendererImp::draw_image( Image& image ) {

  // Advanced Task
  // Render image element with rotation

  Vector2D p0 = transform(image.position);
  Vector2D p1 = transform(image.position + image.dimension);

  rasterize_image( p0.x, p0.y, p1.x, p1.y, image.tex );
}

void SoftwareRendererImp::draw_group( Group& group ) {

  for ( size_t i = 0; i < group.elements.size(); ++i ) {
    draw_element(group.elements[i]);
  }

}

// Rasterization //

// The input arguments in the rasterization functions 
// below are all defined in screen space coordinates

void SoftwareRendererImp::rasterize_point( float x, float y, Color color ) {

  // fill in the nearest pixel
  int sx = (int)floor(x);
  int sy = (int)floor(y);

  // check bounds
  if (sx < 0 || sx >= width) return;
  if (sy < 0 || sy >= height) return;

  // fill sample - NOT doing alpha blending!
  // TODO: Call fill_pixel here to run alpha blending
  fill_pixel(sx, sy, color);
}

void SoftwareRendererImp::rasterize_line( float x0, float y0,
                                          float x1, float y1,
                                          Color color) {

  // Task 0: 
  // Implement Bresenham's algorithm (delete the line below and implement your own)
  // ref->rasterize_line_helper(x0, y0, x1, y1, width, height, color, this);

  // float dx = x1 - x0;
  // float dy = y1 - y0;
  // if (dx == 0 && dy == 0) {
  //   rasterize_point(x0, y0, color);
  // }

  // // handle slope > 1
  // bool large_slope = false;
  // auto rasterize_point_wrapper = [&](float x, float y, Color color, bool flip) {
  //   if (flip) 
  //     rasterize_point(y, x, color);
  //   else 
  //     rasterize_point(x, y, color);
  // };
  // if (abs(dy) > abs(dx)) {
  //   large_slope = true;
  //   std::swap(x0, y0);
  //   std::swap(x1, y1);
  // }

  // // handle third and forth quadrant
  // if (x0 > x1) {
  //   std::swap(x0, x1);
  //   std::swap(y0, y1);
  // }

  // // Turn all vectors into the 1st or the 8th octant
  // // update dx dy accordingly
  // dx = x1 - x0;
  // dy = y1 - y0;
  // float slope = dy / dx;
  
  // // handle two end points
  // rasterize_point_wrapper(x0, y0, color, large_slope);
  // rasterize_point_wrapper(x1, y1, color, large_slope);
  
  // float cx = floor(x0) + 1.5; float ex = floor(x1) - 0.5;
  // float cy = y0 + slope * (cx - x0);
  // while (cx <= ex) {
  //   rasterize_point_wrapper(cx, cy, color, large_slope);
  //   cx++;
  //   cy += slope;
  // }

  // Advanced Task
  // Drawing Smooth Lines with Line Width
  auto ipart = [](float x) -> int {
    return static_cast<int>(x);
  };

  auto round = [&](float x) -> int {
    return ipart(x + 0.5);
  };

  auto fpart = [&](float x) -> float {
    return x - ipart(x);
  };

  auto rfpart = [&](float x) -> float {
    return 1 - fpart(x);
  };

  bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);

  if (steep) {
    std::swap(x0, y0);
    std::swap(x1, y1);
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }

  float dx = x1 - x0;
  float dy = y1 - y0;
  float gradient;

  if (dx == 0) {
    gradient = 1.0;
  } else {
    gradient = dy / dx;
  }

  // Adjusting for half-integer pixel centers
  x0 -= 0.5f;
  y0 -= 0.5f;
  x1 -= 0.5f;
  y1 -= 0.5f;

  // handle first endpoint
  int xend = round(x0);
  float yend = y0 + gradient * (xend - x0);
  float xgap = rfpart(x0 + 0.5);
  int xpxl1 = xend; // this will be used in the main loop
  int ypxl1 = ipart(yend);
  if (steep) {
    rasterize_point(ypxl1, xpxl1, color * rfpart(yend) * xgap);
    rasterize_point(ypxl1 + 1, xpxl1, color * fpart(yend) * xgap);
  } else {
    rasterize_point(xpxl1, ypxl1, color * rfpart(yend) * xgap);
    rasterize_point(xpxl1, ypxl1 + 1, color * fpart(yend) * xgap);
  }
  float intery = yend + gradient; // first y-intersection for the main loop

  // handle second endpoint
  xend = round(x1);
  yend = y1 + gradient * (xend - x1);
  xgap = fpart(x1 + 0.5);
  int xpxl2 = xend; // this will be used in the main loop
  int ypxl2 = ipart(yend);
  if (steep) {
    rasterize_point(ypxl2, xpxl2, color * rfpart(yend) * xgap);
    rasterize_point(ypxl2 + 1, xpxl2, color * fpart(yend) * xgap);
  } else {
    rasterize_point(xpxl2, ypxl2, color * rfpart(yend) * xgap);
    rasterize_point(xpxl2, ypxl2 + 1, color * fpart(yend) * xgap);
  }

  // main loop
  if (steep) {
    for (int x = xpxl1 + 1; x <= xpxl2 - 1; x++) {
      rasterize_point(ipart(intery), x, color * rfpart(intery));
      rasterize_point(ipart(intery) + 1, x, color * fpart(intery));
      intery += gradient;
    }
  } else {
    for (int x = xpxl1 + 1; x <= xpxl2 - 1; x++) {
      rasterize_point(x, ipart(intery), color * rfpart(intery));
      rasterize_point(x, ipart(intery) + 1, color * fpart(intery));
      intery += gradient;
    }
  }
}

inline void initEdge(float x0, float y0, float x1, float y1, float top_y, Edge* l) {
  if (!l) perror("Edge struct not initialized!");
  
  l->A = y1 - y0;
  l->B = x0 - x1;
  l->C = y0 * (x1 - x0) - x0 * (y1 - y0);
  l->isTopLeft = (y0 == y1 == top_y) || (y0 < y1);  // top or left edges
}

inline float testValue(float x, float y, Edge* l) {
  if (!l) perror("Edge struct not initialized!");
  
  return l->A * x + l->B * y + l->C;
}

void SoftwareRendererImp::rasterize_triangle( float x0, float y0,
                                              float x1, float y1,
                                              float x2, float y2,
                                              Color color ) {
  // Task 1: 
  // Implement triangle rasterization

  // check points order, make counterclockwise
  Edge l01;
  float top_y = min(y0, min(y1, y2));
  initEdge(x0, y0, x1, y1, top_y, &l01);
  float test = testValue(x2, y2, &l01);
  if (test > 0) {
    // clockwise, make counter-clockwise for easier processing
    std::swap(x1, x2);
    std::swap(y1, y2);
  }

  // transform to sample space
  x0 *= sample_rate; y0 *= sample_rate;
  x1 *= sample_rate; y1 *= sample_rate;
  x2 *= sample_rate; y2 *= sample_rate;
  size_t s_width = width * sample_rate;
  size_t s_height = height * sample_rate;
  top_y *= sample_rate;

  // calculate edge formula and whether it's a top / left edge
  Edge edges[3];
  initEdge(x0, y0, x1, y1, top_y, &edges[0]);
  initEdge(x1, y1, x2, y2, top_y, &edges[1]);
  initEdge(x2, y2, x0, y0, top_y, &edges[2]);

  // compute bounding box rounding the nearest sample point 
  float min_x = (int)(max(0.f, (min(x0, min(x1, x2))) - 1)) + 0.5f;
  float max_x = (int)(min(s_width - 0.5f, max(x0, max(x1, x2)) + 1)) + 0.5f;
  float min_y = (int)(max(0.f, (min(y0, min(y1, y2))) - 1)) + 0.5f;
  float max_y = (int)(min(s_height - 0.5f, max(y0, max(y1, y2)) + 1)) + 0.5f;

  // incremental triangle traversal in zigzag order starting from the bottom-left point
  float cx = min_x; float cy = min_y;
  float test_vals[3];
  for (int i = 0; i < 3; i++) {
    test_vals[i] = testValue(cx, cy, &edges[i]);
  }
  bool zig = true;

  while (cy <= max_y) {
    int sx = (int) cx;
    int sy = (int) cy;
    int s_idx = sx + sy * s_width;

    for (int i = 0; i < 3; i++) {
      if (test_vals[i] > 0 || (test_vals[i] == 0 && !edges[i].isTopLeft)) goto next_sample;
    }

    fill_sample(sx, sy, color);

next_sample:
    if (zig && cx < max_x) {
      cx++;
      for (int i = 0; i < 3; i++) {
        test_vals[i] += edges[i].A;
      }
    } else if (!zig && cx > min_x) {
      cx--;
      for (int i = 0; i < 3; i++) {
        test_vals[i] -= edges[i].A;
      }
    } else {
      zig = !zig;
      cy++;
      for (int i = 0; i < 3; i++) {
        test_vals[i] += edges[i].B;
      }
    }
  }
  
  // Advanced Task
  // Implementing Triangle Edge Rules
  // Implemented in functions for struct Edge
}

void SoftwareRendererImp::rasterize_image( float x0, float y0,
                                           float x1, float y1,
                                           Texture& tex ) {
  // Task 4: 
  // Implement image rasterization
  Sampler2DImp sampler;

  int minX = static_cast<int>(std::floor(x0));
  int maxX = static_cast<int>(std::floor(x1));
  int minY = static_cast<int>(std::floor(y0));
  int maxY = static_cast<int>(std::floor(y1));

  float scaleX = 1.0f / (x1 - x0);
  float scaleY = 1.0f / (y1 - y0);

  for (int x = minX; x <= maxX; x++) {
    for (int y = minY; y <= maxY; y++) {
      if (x < 0 || x >= width || y < 0 || y >= height) continue;

      float u = (x + 0.5f - x0) * scaleX;
      float v = (y + 0.5f - y0) * scaleY;

      if ((u < 0 || u > 1) || (v < 0 || v > 1)) continue;

      Color c = sampler.sample_bilinear(tex, u, v);
      
      fill_pixel(x, y, c);
    }
  }
}

// resolve samples to pixel buffer
void SoftwareRendererImp::resolve( void ) {
  // Task 2: 
  // Implement supersampling
  // You may also need to modify other functions marked with "Task 2".
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int base_i = i * sample_rate;
      int base_j = j * sample_rate;

      unsigned r = 0; unsigned g = 0; unsigned b = 0; unsigned a = 0;
      for (int ii = 0; ii < sample_rate; ii++) {
        for (int jj = 0; jj < sample_rate; jj++) {
          int s_idx = (base_j + jj) * sample_rate * width + base_i + ii;       
          r += (unsigned)sample_buffer[s_idx * 4];
          g += (unsigned)sample_buffer[s_idx * 4 + 1];
          b += (unsigned)sample_buffer[s_idx * 4 + 2];
          a += (unsigned)sample_buffer[s_idx * 4 + 3];
        }
      }
      
      int p_idx = j * width + i;
      size_t sample_per_pixel = sample_rate * sample_rate;

      pixel_buffer[p_idx * 4] = r / sample_per_pixel;
      pixel_buffer[p_idx * 4 + 1] = g / sample_per_pixel;
      pixel_buffer[p_idx * 4 + 2] = b / sample_per_pixel;
      pixel_buffer[p_idx * 4 + 3] = a / sample_per_pixel;
    }
  }

  clear_sample_buffer();
}

Color SoftwareRendererImp::alpha_blending(Color pixel_color, Color color)
{
  // Task 5
  // Implement alpha compositing
  pixel_color.r = (1 - color.a) * pixel_color.r + color.r;
  pixel_color.g = (1 - color.a) * pixel_color.g + color.g;
  pixel_color.b = (1 - color.a) * pixel_color.b + color.b;
  pixel_color.a = 1 - (1 - color.a) * (1 - pixel_color.a);

  return pixel_color;
}


} // namespace CS248
