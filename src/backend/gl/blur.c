#include <locale.h>
#include <stdbool.h>

#include "backend/backend_common.h"
#include "utils/misc.h"

#include "gl_common.h"

struct gl_blur_context {
	struct gl_shader *blur_shader;

	/// Temporary textures used for blurring
	GLuint *blur_textures;
	/// Temporary fbos used for blurring
	GLuint *blur_fbos;

	int blur_texture_count;
	int blur_fbo_count;

	/// Cached dimensions of each blur_texture. They keep growing to accommodate each
	/// source image it gets. The hope is soon it gets big enough and we stop
	/// resizing.
	/// Turns out calling glTexImage to resize is expensive, so we avoid that.
	struct texture_size {
		int width;
		int height;
	} *texture_sizes;

	/// Final render target size, used to detect root size change and triggers
	/// blur_texture shrinking to reduce memory usage.
	uint64_t root_size;

	/// Cached dimensions of the offscreen framebuffer. It's the same size as the
	/// target but is expanded in either direction by resize_width / resize_height.
	int fb_width, fb_height;

	/// How much do we need to resize the damaged region for blurring.
	int resize_width, resize_height;

	int noise_radius;
	float noise_scale;
	/// Maximum of the focused/unfocused liquid glass strengths, used for
	/// sizing damage regions.
	int liquid_glass_strength;
	int liquid_glass_distance;
	int dual_kawase_strength;
	float liquid_glass_curve[4];

	int npasses;

	enum backend_image_format format;

	enum blur_method method;
};

static void gl_set_blur_gradient_uniforms(const struct backend_blur_args *args,
                                          ivec2 origin, bool enabled) {
	glUniform2f(UNIFORM_BLUR_GRADIENT_START_LOC,
	            (GLfloat)(args->blur_gradient_start.x - origin.x),
	            (GLfloat)(args->blur_gradient_start.y - origin.y));
	glUniform2f(UNIFORM_BLUR_GRADIENT_END_LOC,
	            (GLfloat)(args->blur_gradient_end.x - origin.x),
	            (GLfloat)(args->blur_gradient_end.y - origin.y));
	GLfloat points[BLUR_GRADIENT_MAX_POINTS * 2];
	for (unsigned int i = 0; i < args->blur_gradient_point_count; i++) {
		points[i * 2] = (GLfloat)args->blur_gradient_points[i].position;
		points[i * 2 + 1] = (GLfloat)args->blur_gradient_points[i].amount;
	}
	glUniform2fv(UNIFORM_BLUR_GRADIENT_POINTS_LOC,
	             (GLsizei)args->blur_gradient_point_count, points);
	glUniform1i(UNIFORM_BLUR_GRADIENT_POINT_COUNT_LOC,
	            (GLint)args->blur_gradient_point_count);
	glUniform1i(UNIFORM_BLUR_GRADIENT_ENABLED_LOC, enabled ? 1 : 0);
}

// TODO(yshui) small optimization for kernel blur, if source and target are different,
// single pass blur can paint directly from source to target. Currently a temporary
// texture is always used.

/**
 * Blur contents in a particular region.
 */
static bool
gl_kernel_blur(double opacity, struct gl_blur_context *bctx,
               const struct backend_mask_image *mask, vec2 mask_scale,
               const GLuint vao[2], const int vao_nelems[2], struct gl_texture *source,
               GLuint blur_sampler, GLuint target_fbo, GLuint default_mask) {
	int curr = 0;
	for (int i = 0; i < bctx->npasses; ++i) {
		auto p = &bctx->blur_shader[i];
		assert(p->prog);

		assert(bctx->blur_textures[curr]);

		// The origin to use when sampling from the source texture
		GLint tex_width, tex_height;
		GLuint src_texture;

		if (i == 0) {
			src_texture = source->texture;
			tex_width = source->width;
			tex_height = source->height;
		} else {
			src_texture = bctx->blur_textures[curr];
			auto src_size = bctx->texture_sizes[curr];
			tex_width = src_size.width;
			tex_height = src_size.height;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, src_texture);
		glBindSampler(0, blur_sampler);
		glUseProgram(p->prog);
		glUniform2f(UNIFORM_PIXEL_NORM_LOC, 1.0F / (GLfloat)tex_width,
		            1.0F / (GLfloat)tex_height);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, default_mask);

		glUniform1i(UNIFORM_MASK_TEX_LOC, 1);
		glUniform2f(UNIFORM_MASK_OFFSET_LOC, 0.0F, 0.0F);
		glUniform1i(UNIFORM_MASK_INVERTED_LOC, 0);
		glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, 0.0F);
		glUniform2f(UNIFORM_MASK_SCALE_LOC, 1.0F, 1.0F);

		// The number of indices in the selected vertex array
		GLsizei nelems;

		if (i < bctx->npasses - 1) {
			assert(bctx->blur_fbos[0]);
			assert(bctx->blur_textures[!curr]);

			// not last pass, draw into framebuffer, with resized regions
			glBindVertexArray(vao[1]);
			nelems = vao_nelems[1];
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, bctx->blur_fbos[0]);

			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			                       GL_TEXTURE_2D, bctx->blur_textures[!curr], 0);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);
			if (!gl_check_fb_complete(GL_FRAMEBUFFER)) {
				return false;
			}

			glUniform1f(UNIFORM_OPACITY_LOC, 1.0F);
			glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, 0);
			glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, 0.0F);
		} else {
			// last pass, draw directly into the back buffer, with origin
			// regions. And apply mask if requested
			if (mask != NULL) {
				auto inner = (struct gl_texture *)mask->image;
				log_trace("Mask texture is %d", inner->texture);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, inner->texture);
				glUniform1i(UNIFORM_MASK_INVERTED_LOC, mask->inverted);
				glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC,
				            (float)mask->corner_radius);
				glUniform2f(UNIFORM_MASK_OFFSET_LOC,
				            (float)mask->origin.x, (float)mask->origin.y);
				glUniform2f(UNIFORM_MASK_SCALE_LOC, (float)mask_scale.x,
				            (float)mask_scale.y);
			}
			glBindVertexArray(vao[0]);
			nelems = vao_nelems[0];
			glBindFramebuffer(GL_FRAMEBUFFER, target_fbo);

			glUniform1f(UNIFORM_OPACITY_LOC, (float)opacity);
			glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, bctx->noise_radius);
			glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, bctx->noise_scale);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		}

		glDrawElements(GL_TRIANGLES, nelems, GL_UNSIGNED_INT, NULL);

		// XXX use multiple draw calls is probably going to be slow than
		//     just simply blur the whole area.

		curr = !curr;
	}

	return true;
}

/// Do dual-kawase blur.
///
/// @param vao two vertex array objects.
///            [0]: for sampling from blurred result into the target fbo.
///            [1]: for sampling from the source texture into blurred textures.
bool gl_dual_kawase_blur(ivec2 origin, double opacity, float dual_kawase_scale, int kawase_strength,
                         int glass_strength,
                          int glass_distance, int glass_aberration, float glass_zoom,
                          float glass_saturation, vec2 glass_anchor,
                          bool glass_zoom_follow_mouse, bool no_dual_kawase,
                          const struct backend_blur_args *args,
                          struct gl_blur_context *bctx,
                         const struct backend_mask_image *mask, vec2 mask_scale,
                         const GLuint vao[2], const int vao_nelems[2],
                         struct gl_texture *source, GLuint blur_sampler, GLuint target_fbo,
                          GLuint default_mask) {
	static const int strength_iterations[20] = {
	    1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};
	static const float strength_offsets[20] = {
	    1.25F, 2.25F, 2.00F, 3.00F, 4.25F, 2.50F, 3.25F, 4.25F, 5.50F, 3.25F,
	    4.00F, 5.00F, 6.00F, 7.25F, 8.25F, 4.50F, 5.25F, 6.25F, 7.25F, 8.50F};
	int selected_strength = kawase_strength > 0 ? kawase_strength : bctx->dual_kawase_strength;
	selected_strength = clamp(selected_strength, 1, 20);
	int iterations = strength_iterations[selected_strength - 1];
	iterations = min2(iterations, bctx->blur_texture_count);
	float strength_scale = dual_kawase_scale;
	if (bctx->dual_kawase_strength > 0) {
		strength_scale *= strength_offsets[selected_strength - 1] /
		                  strength_offsets[bctx->dual_kawase_strength - 1];
	}
	int scale_factor = 1;

	if (no_dual_kawase && bctx->method == BLUR_METHOD_LIQUID_GLASS) {
		// Use the original source for liquid-glass displacement and aberration.
		auto up_pass = &bctx->blur_shader[1];
		assert(up_pass->prog);
		glUseProgram(up_pass->prog);
		// A zero sampling offset turns the liquid-glass filter into a direct
		// source sample while preserving refraction and aberration offsets.
		glUniform1f(UNIFORM_DUAL_KAWASE_SCALE_LOC, 0.0F);
		glUniform1f(UNIFORM_LIQUID_GLASS_SOURCE_SCALE_LOC, 1.0F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, source->texture);
		glBindSampler(0, blur_sampler);
		glActiveTexture(GL_TEXTURE1);
		if (mask != NULL) {
			auto inner = (struct gl_texture *)mask->image;
			glBindTexture(GL_TEXTURE_2D, inner->texture);
			glUniform1i(UNIFORM_MASK_INVERTED_LOC, mask->inverted);
			glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, (float)mask->corner_radius);
			glUniform2f(UNIFORM_MASK_OFFSET_LOC, (float)mask->origin.x,
			            (float)mask->origin.y);
			glUniform2f(UNIFORM_MASK_SCALE_LOC, (float)mask_scale.x,
			            (float)mask_scale.y);
		} else {
			glBindTexture(GL_TEXTURE_2D, default_mask);
			glUniform1i(UNIFORM_MASK_INVERTED_LOC, 0);
			glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, 0.0F);
			glUniform2f(UNIFORM_MASK_OFFSET_LOC, 0.0F, 0.0F);
			glUniform2f(UNIFORM_MASK_SCALE_LOC, 1.0F, 1.0F);
		}
		glUniform1i(UNIFORM_MASK_TEX_LOC, 1);
		glUniform1f(UNIFORM_OPACITY_LOC, (float)opacity);
		gl_set_blur_gradient_uniforms(args, origin, false);
		glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, bctx->noise_radius);
		glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, bctx->noise_scale);
		glUniform1i(UNIFORM_LIQUID_GLASS_STRENGTH_LOC, glass_strength);
		glUniform1f(UNIFORM_LIQUID_GLASS_ABERRATION_LOC, (float)glass_aberration);
		glUniform1i(UNIFORM_LIQUID_GLASS_DISTANCE_LOC, glass_distance);
		glUniform1f(UNIFORM_LIQUID_GLASS_ZOOM_LOC, glass_zoom);
		glUniform1f(UNIFORM_LIQUID_GLASS_SATURATION_LOC, glass_saturation);
		glUniform1i(UNIFORM_LIQUID_GLASS_ANCHOR_MOUSE_LOC,
		            glass_zoom_follow_mouse ? 1 : 0);
		glUniform2f(UNIFORM_LIQUID_GLASS_ANCHOR_LOC, (GLfloat)glass_anchor.x,
		            (GLfloat)glass_anchor.y);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X1_LOC, bctx->liquid_glass_curve[0]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y1_LOC, bctx->liquid_glass_curve[1]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X2_LOC, bctx->liquid_glass_curve[2]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y2_LOC, bctx->liquid_glass_curve[3]);
		glUniform1f(UNIFORM_SCALE_LOC, 1.0F);
		glUniform2f(UNIFORM_PIXEL_NORM_LOC, 1.0F / (GLfloat)source->width,
		            1.0F / (GLfloat)source->height);
		glBindVertexArray(vao[0]);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glDrawElements(GL_TRIANGLES, vao_nelems[0], GL_UNSIGNED_INT, NULL);
		return true;
	}

	// Kawase downsample pass
	auto down_pass = &bctx->blur_shader[0];
	assert(down_pass->prog);
	glUseProgram(down_pass->prog);
	glUniform1f(UNIFORM_DUAL_KAWASE_SCALE_LOC, strength_scale);

	glBindVertexArray(vao[1]);
	int nelems = vao_nelems[1];

	for (int i = 0; i < iterations; ++i) {
		// Scale output width / height by half in each iteration
		scale_factor <<= 1;

		GLuint src_texture;
		int tex_width, tex_height;

		if (i == 0) {
			// first pass: copy from back buffer
			src_texture = source->texture;
			tex_width = source->width;
			tex_height = source->height;
		} else {
			// copy from previous pass
			src_texture = bctx->blur_textures[i - 1];
			auto src_size = bctx->texture_sizes[i - 1];
			tex_width = src_size.width;
			tex_height = src_size.height;
		}

		assert(src_texture);
		assert(bctx->blur_fbos[i]);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, src_texture);
		glBindSampler(0, blur_sampler);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, bctx->blur_fbos[i]);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);

		glUniform1f(UNIFORM_SCALE_LOC, (GLfloat)scale_factor);

		glUniform2f(UNIFORM_PIXEL_NORM_LOC, 1.0F / (GLfloat)tex_width,
		            1.0F / (GLfloat)tex_height);

		glDrawElements(GL_TRIANGLES, nelems, GL_UNSIGNED_INT, NULL);
	}

	// Kawase upsample pass
	auto up_pass = &bctx->blur_shader[1];
	assert(up_pass->prog);
	glUseProgram(up_pass->prog);
	glUniform1f(UNIFORM_DUAL_KAWASE_SCALE_LOC, strength_scale);
	glUniform1f(UNIFORM_LIQUID_GLASS_SOURCE_SCALE_LOC, 0.5F);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, default_mask);

	glUniform1i(UNIFORM_MASK_TEX_LOC, 1);
	glUniform2f(UNIFORM_MASK_OFFSET_LOC, 0.0F, 0.0F);
	glUniform1i(UNIFORM_MASK_INVERTED_LOC, 0);
	glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, 0.0F);
	glUniform2f(UNIFORM_MASK_SCALE_LOC, 1.0F, 1.0F);
	glUniform1f(UNIFORM_OPACITY_LOC, 1.0F);
	glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, 0);
	glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, 0.0F);
	glUniform1i(UNIFORM_LIQUID_GLASS_STRENGTH_LOC, 0);
	glUniform1f(UNIFORM_LIQUID_GLASS_ABERRATION_LOC, 0.0F);
	// The liquid-glass shader is reused for intermediate upsample passes.
	// Keep its zoom neutral until the final compositing pass below.
	glUniform1f(UNIFORM_LIQUID_GLASS_ZOOM_LOC, 1.0F);
	glUniform1f(UNIFORM_LIQUID_GLASS_SATURATION_LOC, 1.0F);
	glUniform1i(UNIFORM_LIQUID_GLASS_ANCHOR_MOUSE_LOC, 0);
	glUniform2f(UNIFORM_LIQUID_GLASS_ANCHOR_LOC, 0.0F, 0.0F);
	glUniform1i(UNIFORM_BLUR_GRADIENT_ENABLED_LOC, 0);
	glUniform1i(UNIFORM_BLUR_GRADIENT_POINT_COUNT_LOC, 0);

	if (args->blur_gradient_enabled && !no_dual_kawase) {
		// The gradient needs all downsampled levels at once. Keep the pyramid
		// intact and let the final shader interpolate between its levels locally.
		glUseProgram(up_pass->prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, bctx->blur_textures[iterations - 1]);
		glBindSampler(0, blur_sampler);
		glUniform1f(UNIFORM_SCALE_LOC, 1.0F);
		glUniform2f(UNIFORM_PIXEL_NORM_LOC,
		            1.0F / (GLfloat)bctx->texture_sizes[0].width,
		            1.0F / (GLfloat)bctx->texture_sizes[0].height);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, source->texture);
		glBindSampler(2, blur_sampler);
		glUniform1i(UNIFORM_LIQUID_GLASS_SOURCE_TEX_LOC, 2);
		glUniform1i(UNIFORM_BLUR_GRADIENT_LEVEL_COUNT_LOC, iterations);
		for (int level = 0; level < 5; level++) {
			const int unit = 3 + level;
			const int texture_level = min2(level, iterations - 1);
			glActiveTexture((GLenum)(GL_TEXTURE0 + unit));
			glBindTexture(GL_TEXTURE_2D, bctx->blur_textures[texture_level]);
			glBindSampler((GLuint)unit, blur_sampler);
			glUniform1i(UNIFORM_BLUR_GRADIENT_LEVEL_0_LOC + level, unit);
		}
		glActiveTexture(GL_TEXTURE1);
		if (mask != NULL) {
			auto inner = (struct gl_texture *)mask->image;
			glBindTexture(GL_TEXTURE_2D, inner->texture);
			glUniform1i(UNIFORM_MASK_INVERTED_LOC, mask->inverted);
			glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, (float)mask->corner_radius);
			glUniform2f(UNIFORM_MASK_OFFSET_LOC, (float)mask->origin.x,
			            (float)mask->origin.y);
			glUniform2f(UNIFORM_MASK_SCALE_LOC, (float)mask_scale.x,
			            (float)mask_scale.y);
		} else {
			glBindTexture(GL_TEXTURE_2D, default_mask);
			glUniform1i(UNIFORM_MASK_INVERTED_LOC, 0);
			glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC, 0.0F);
			glUniform2f(UNIFORM_MASK_OFFSET_LOC, 0.0F, 0.0F);
			glUniform2f(UNIFORM_MASK_SCALE_LOC, 1.0F, 1.0F);
		}
		glUniform1i(UNIFORM_MASK_TEX_LOC, 1);
		glUniform1f(UNIFORM_OPACITY_LOC, (GLfloat)opacity);
		glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, bctx->noise_radius);
		glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, bctx->noise_scale);
		glUniform1i(UNIFORM_LIQUID_GLASS_STRENGTH_LOC, glass_strength);
		glUniform1f(UNIFORM_LIQUID_GLASS_ABERRATION_LOC, (float)glass_aberration);
		glUniform1i(UNIFORM_LIQUID_GLASS_DISTANCE_LOC, glass_distance);
		glUniform1f(UNIFORM_LIQUID_GLASS_ZOOM_LOC, glass_zoom);
		glUniform1f(UNIFORM_LIQUID_GLASS_SATURATION_LOC, glass_saturation);
		glUniform1i(UNIFORM_LIQUID_GLASS_ANCHOR_MOUSE_LOC,
		            glass_zoom_follow_mouse ? 1 : 0);
		glUniform2f(UNIFORM_LIQUID_GLASS_ANCHOR_LOC, (GLfloat)glass_anchor.x,
		            (GLfloat)glass_anchor.y);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X1_LOC, bctx->liquid_glass_curve[0]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y1_LOC, bctx->liquid_glass_curve[1]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X2_LOC, bctx->liquid_glass_curve[2]);
		glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y2_LOC, bctx->liquid_glass_curve[3]);
		gl_set_blur_gradient_uniforms(args, origin, true);
		glBindVertexArray(vao[0]);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glDrawElements(GL_TRIANGLES, vao_nelems[0], GL_UNSIGNED_INT, NULL);
		glActiveTexture(GL_TEXTURE0);
		return true;
	}

	for (int i = iterations - 1; i >= 0; --i) {
		// Scale output width / height back by two in each iteration
		scale_factor >>= 1;

		const GLuint src_texture = bctx->blur_textures[i];
		assert(src_texture);

		// Calculate normalized half-width/-height of a src pixel
		auto src_size = bctx->texture_sizes[i];
		int tex_width = src_size.width;
		int tex_height = src_size.height;

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, src_texture);
		glBindSampler(0, blur_sampler);

		if (i > 0) {
			assert(bctx->blur_fbos[i - 1]);

			// not last pass, draw into next framebuffer
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, bctx->blur_fbos[i - 1]);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);
		} else {
			// last pass, draw directly into the target fbo
			if (mask != NULL) {
				auto inner = (struct gl_texture *)mask->image;
				log_trace("Mask texture is %d", inner->texture);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, inner->texture);
				glUniform1i(UNIFORM_MASK_INVERTED_LOC, mask->inverted);
				glUniform1f(UNIFORM_MASK_CORNER_RADIUS_LOC,
				            (float)mask->corner_radius);
				glUniform2f(UNIFORM_MASK_OFFSET_LOC, (float)(mask->origin.x),
				            (float)(mask->origin.y));
				glUniform2f(UNIFORM_MASK_SCALE_LOC, (float)mask_scale.x,
				            (float)mask_scale.y);
			}
			glBindVertexArray(vao[0]);
			nelems = vao_nelems[0];
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo);
			glUniform1f(UNIFORM_OPACITY_LOC, (GLfloat)opacity);
			gl_set_blur_gradient_uniforms(args, origin, false);
			glUniform1i(UNIFORM_BLUR_NOISE_RADIUS_LOC, bctx->noise_radius);
			glUniform1f(UNIFORM_BLUR_NOISE_SCALE_LOC, bctx->noise_scale);
			glUniform1i(UNIFORM_LIQUID_GLASS_STRENGTH_LOC, glass_strength);
			glUniform1f(UNIFORM_LIQUID_GLASS_ABERRATION_LOC, (float)glass_aberration);
			glUniform1i(UNIFORM_LIQUID_GLASS_DISTANCE_LOC, glass_distance);
			glUniform1f(UNIFORM_LIQUID_GLASS_ZOOM_LOC, glass_zoom);
			glUniform1f(UNIFORM_LIQUID_GLASS_SATURATION_LOC, glass_saturation);
			glUniform1i(UNIFORM_LIQUID_GLASS_ANCHOR_MOUSE_LOC,
			            glass_zoom_follow_mouse ? 1 : 0);
			glUniform2f(UNIFORM_LIQUID_GLASS_ANCHOR_LOC,
			            (GLfloat)glass_anchor.x, (GLfloat)glass_anchor.y);
			glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X1_LOC, bctx->liquid_glass_curve[0]);
			glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y1_LOC, bctx->liquid_glass_curve[1]);
			glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_X2_LOC, bctx->liquid_glass_curve[2]);
			glUniform1f(UNIFORM_LIQUID_GLASS_CURVE_Y2_LOC, bctx->liquid_glass_curve[3]);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		}

		glUniform1f(UNIFORM_SCALE_LOC, (GLfloat)scale_factor);
		glUniform2f(UNIFORM_PIXEL_NORM_LOC, 1.0F / (GLfloat)tex_width,
		            1.0F / (GLfloat)tex_height);

		glDrawElements(GL_TRIANGLES, nelems, GL_UNSIGNED_INT, NULL);
	}

	return true;
}

static bool gl_blur_context_preallocate_textures(struct gl_blur_context *bctx,
                                                 ivec2 source_size, uint64_t root_size) {
	if (root_size != bctx->root_size) {
		bctx->root_size = root_size;
		bctx->fb_width = 0;
		bctx->fb_height = 0;
	}

	if (source_size.width > bctx->fb_width || source_size.height > bctx->fb_height) {
		// Resize the temporary textures used for blur in case the source image
		// can't fit.
		bctx->fb_width = max2(bctx->fb_width, source_size.width);
		bctx->fb_height = max2(bctx->fb_height, source_size.height);

		for (int i = 0; i < bctx->blur_texture_count; ++i) {
			auto tex_size = bctx->texture_sizes + i;
			if (bctx->method == BLUR_METHOD_DUAL_KAWASE ||
			    bctx->method == BLUR_METHOD_LIQUID_GLASS) {
				// Use smaller textures for each iteration (quarter of the
				// previous texture)
				tex_size->width = 1 + ((bctx->fb_width - 1) >> (i + 1));
				tex_size->height = 1 + ((bctx->fb_height - 1) >> (i + 1));
			} else {
				tex_size->width = bctx->fb_width;
				tex_size->height = bctx->fb_height;
			}

			glBindTexture(GL_TEXTURE_2D, bctx->blur_textures[i]);
			GLint format;
			switch (bctx->format) {
			case BACKEND_IMAGE_FORMAT_PIXMAP_HIGH: format = GL_RGBA16; break;
			case BACKEND_IMAGE_FORMAT_PIXMAP: format = GL_RGBA8; break;
			case BACKEND_IMAGE_FORMAT_MASK: format = GL_R8; break;
			default: unreachable();
			}
			glTexImage2D(GL_TEXTURE_2D, 0, format, tex_size->width,
			             tex_size->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

			if (bctx->method == BLUR_METHOD_DUAL_KAWASE ||
			    bctx->method == BLUR_METHOD_LIQUID_GLASS) {
				// Attach texture to FBO target
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, bctx->blur_fbos[i]);
				glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
				                       GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
				                       bctx->blur_textures[i], 0);
				if (!gl_check_fb_complete(GL_FRAMEBUFFER)) {
					glBindFramebuffer(GL_FRAMEBUFFER, 0);
					return false;
				}
			}
		}
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	}
	return true;
}

bool gl_blur(struct backend_base *base, ivec2 origin, image_handle target_,
             const struct backend_blur_args *args) {
	auto gd = (struct gl_data *)base;
	auto target = (struct gl_texture *)target_;
	auto source = (struct gl_texture *)args->source_image;
	auto bctx = (struct gl_blur_context *)args->blur_context;
	log_trace("Blur size: %dx%d, method: %d", source->width, source->height, bctx->method);
	bool ret = false;

	// Remainder: regions are in Xorg coordinates
	auto reg_blur_resized =
	    resize_region(args->target_mask, bctx->resize_width, bctx->resize_height);
	const rect_t *extent = pixman_region32_extents(args->target_mask);
	int width = extent->x2 - extent->x1, height = extent->y2 - extent->y1;
	if (width == 0 || height == 0) {
		return true;
	}

	int nrects, nrects_resized;
	const rect_t *rects = pixman_region32_rectangles(args->target_mask, &nrects),
	             *rects_resized =
	                 pixman_region32_rectangles(&reg_blur_resized, &nrects_resized);
	if (!nrects || !nrects_resized) {
		return true;
	}

	uint64_t root_size = (uint64_t)gd->back_image.height * (uint64_t)gd->back_image.width;
	if (!gl_blur_context_preallocate_textures(
	        bctx, (ivec2){source->width, source->height}, root_size)) {
		return false;
	}

	// Original region for the final compositing step from blur result to target.
	auto coord = ccalloc(nrects * 16, GLfloat);
	auto indices = ccalloc(nrects * 6, GLuint);
	gl_mask_rects_to_coords(origin, nrects, rects, SCALE_IDENTITY, coord, indices);
	if (!target->y_inverted) {
		gl_y_flip_target(nrects, coord, target->height);
	}

	// Resize region for sampling from source texture, and for blur passes
	auto coord_resized = ccalloc(nrects_resized * 16, GLfloat);
	auto indices_resized = ccalloc(nrects_resized * 6, GLuint);
	gl_mask_rects_to_coords(origin, nrects_resized, rects_resized, SCALE_IDENTITY,
	                        coord_resized, indices_resized);
	pixman_region32_fini(&reg_blur_resized);
	// FIXME(yshui) In theory we should handle blurring a non-y-inverted source, but
	// we never actually use that capability anywhere.
	assert(source->y_inverted);

	glBlendFunc(GL_ONE, GL_ZERO);
	glBindVertexArray(gd->vertex_array_objects[0]);
	glBindBuffer(GL_ARRAY_BUFFER, gd->buffer_objects[0]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gd->buffer_objects[1]);
	glBufferData(GL_ARRAY_BUFFER, (long)sizeof(*coord) * nrects * 16, coord, GL_STREAM_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (long)sizeof(*indices) * nrects * 6,
	             indices, GL_STREAM_DRAW);
	glEnableVertexAttribArray(vert_coord_loc);
	glEnableVertexAttribArray(vert_in_texcoord_loc);
	glVertexAttribPointer(vert_coord_loc, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, NULL);
	glVertexAttribPointer(vert_in_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
	                      sizeof(GLfloat) * 4, (void *)(sizeof(GLfloat) * 2));

	glBindVertexArray(gd->vertex_array_objects[1]);
	glBindBuffer(GL_ARRAY_BUFFER, gd->buffer_objects[2]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gd->buffer_objects[3]);
	glBufferData(GL_ARRAY_BUFFER, (long)sizeof(*coord_resized) * nrects_resized * 16,
	             coord_resized, GL_STREAM_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	             (long)sizeof(*indices_resized) * nrects_resized * 6, indices_resized,
	             GL_STREAM_DRAW);
	glEnableVertexAttribArray(vert_coord_loc);
	glEnableVertexAttribArray(vert_in_texcoord_loc);
	glVertexAttribPointer(vert_coord_loc, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, NULL);
	glVertexAttribPointer(vert_in_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
	                      sizeof(GLfloat) * 4, (void *)(sizeof(GLfloat) * 2));

	int vao_nelems[2] = {nrects * 6, nrects_resized * 6};

	auto target_fbo = gl_bind_image_to_fbo(gd, (image_handle)target);
	if (bctx->method == BLUR_METHOD_DUAL_KAWASE ||
	    bctx->method == BLUR_METHOD_LIQUID_GLASS) {
		ret = gl_dual_kawase_blur(
		    origin, args->opacity, (float)args->dual_kawase_scale, args->dual_kawase_strength,
		    args->liquid_glass_strength,
		    args->liquid_glass_distance,
			args->liquid_glass_aberration, (float)args->liquid_glass_zoom,
		    (float)args->liquid_glass_saturation,
		    args->liquid_glass_anchor,
		    args->liquid_glass_zoom_follow_mouse, args->liquid_glass_no_dual_kawase,
		    args, bctx,
		    args->source_mask,
		    args->source_mask_scale, gd->vertex_array_objects, vao_nelems, source,
		    gd->samplers[GL_SAMPLER_BLUR], target_fbo, gd->default_mask_texture);
	} else {
		ret = gl_kernel_blur(args->opacity, bctx, args->source_mask,
		                     args->source_mask_scale, gd->vertex_array_objects,
		                     vao_nelems, source, gd->samplers[GL_SAMPLER_BLUR],
		                     target_fbo, gd->default_mask_texture);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	for (GLuint unit = 1; unit <= 7; unit++) {
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindSampler(unit, 0);
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Invalidate buffer data
	glBindBuffer(GL_ARRAY_BUFFER, gd->buffer_objects[0]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gd->buffer_objects[1]);
	glBufferData(GL_ARRAY_BUFFER, (long)sizeof(*coord) * nrects * 16, NULL, GL_STREAM_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (long)sizeof(*indices) * nrects * 6, NULL,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, gd->buffer_objects[2]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gd->buffer_objects[3]);
	glBufferData(GL_ARRAY_BUFFER, (long)sizeof(*coord_resized) * nrects_resized * 16,
	             NULL, GL_STREAM_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	             (long)sizeof(*indices_resized) * nrects_resized * 6, NULL,
	             GL_STREAM_DRAW);

	// Cleanup vertex array states
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);

	free(indices);
	free(coord);
	free(indices_resized);
	free(coord_resized);

	gl_check_err();
	return ret;
}

static inline void gl_free_blur_shader(struct gl_shader *shader) {
	if (shader->prog) {
		glDeleteProgram(shader->prog);
	}

	shader->prog = 0;
}

void gl_destroy_blur_context(backend_t *base attr_unused, void *ctx) {
	auto bctx = (struct gl_blur_context *)ctx;
	// Free GLSL shaders/programs
	for (int i = 0; i < bctx->npasses; ++i) {
		gl_free_blur_shader(&bctx->blur_shader[i]);
	}
	free(bctx->blur_shader);

	if (bctx->blur_texture_count && bctx->blur_textures) {
		glDeleteTextures(bctx->blur_texture_count, bctx->blur_textures);
		free(bctx->blur_textures);
	}
	if (bctx->blur_texture_count && bctx->texture_sizes) {
		free(bctx->texture_sizes);
	}
	if (bctx->blur_fbo_count && bctx->blur_fbos) {
		glDeleteFramebuffers(bctx->blur_fbo_count, bctx->blur_fbos);
		free(bctx->blur_fbos);
	}

	bctx->blur_texture_count = 0;
	bctx->blur_fbo_count = 0;

	free(bctx);

	gl_check_err();
}

/**
 * Initialize GL blur filters.
 */
bool gl_create_kernel_blur_context(void *blur_context, GLfloat *projection,
                                   enum blur_method method, void *args) {
	bool success = false;
	auto ctx = (struct gl_blur_context *)blur_context;

	struct conv **kernels;

	int nkernels;
	ctx->method = BLUR_METHOD_KERNEL;
	if (method == BLUR_METHOD_KERNEL) {
		nkernels = ((struct kernel_blur_args *)args)->kernel_count;
		kernels = ((struct kernel_blur_args *)args)->kernels;
	} else {
		kernels = generate_blur_kernel(method, args, &nkernels);
	}

	if (!nkernels) {
		ctx->method = BLUR_METHOD_NONE;
		return true;
	}

	// Specify required textures and FBOs
	ctx->blur_texture_count = 2;
	ctx->blur_fbo_count = 1;

	ctx->blur_shader = ccalloc(max2(2, nkernels), struct gl_shader);

	char *lc_numeric_old = strdup(setlocale(LC_NUMERIC, NULL));
	// Enforce LC_NUMERIC locale "C" here to make sure decimal point is sane
	// Thanks to hiciu for reporting.
	setlocale(LC_NUMERIC, "C");

	// clang-format off
	static const char *FRAG_SHADER_BLUR = GLSL(330,
		layout(location = UNIFORM_TEX_SRC_LOC)
		uniform sampler2D tex_src;
		layout(location = UNIFORM_PIXEL_NORM_LOC)
		uniform vec2 pixel_norm;
		layout(location = UNIFORM_OPACITY_LOC)
		uniform float opacity;
		layout(location = UNIFORM_BLUR_NOISE_RADIUS_LOC)
		uniform int noise_radius;
		layout(location = UNIFORM_BLUR_NOISE_SCALE_LOC)
		uniform float noise_scale;
		in vec2 texcoord;
		out vec4 out_color;
		float mask_factor();
		vec2 perturb(vec2, float, float);
		void main() {
			vec2 uv = perturb(texcoord, noise_radius, noise_scale) * pixel_norm;
			vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
			%s //body of the convolution
			out_color = sum / float(%.7g) * opacity * mask_factor();
		}
	);
	static const char *FRAG_SHADER_BLUR_ADD = QUOTE(
		sum += float(%.7g) * texture2D(tex_src, uv + pixel_norm * vec2(%.7g, %.7g));
	);
	// clang-format on

	const char *shader_add = FRAG_SHADER_BLUR_ADD;

	for (int i = 0; i < nkernels; i++) {
		auto kern = kernels[i];
		// Build shader
		int width = kern->w, height = kern->h;
		int nele = width * height;
		// '%.7g' is at most 14 characters, inserted 3 times
		size_t body_len = (strlen(shader_add) + 42) * (uint)nele;
		char *shader_body = ccalloc(body_len, char);
		char *pc = shader_body;

		// Make use of the linear interpolation hardware by sampling 2 pixels with
		// one texture access by sampling between both pixels based on their
		// relative weight. Easiest done in a single dimension as 2D bilinear
		// filtering would raise additional constraints on the kernels. Therefore
		// only use interpolation along the larger dimension.
		double sum = 0.0;
		if (width > height) {
			// use interpolation in x dimension (width)
			for (int j = 0; j < height; ++j) {
				for (int k = 0; k < width; k += 2) {
					double val1, val2;
					val1 = kern->data[j * width + k];
					val2 = (k + 1 < width)
					           ? kern->data[j * width + k + 1]
					           : 0;

					double combined_weight = val1 + val2;
					if (combined_weight == 0) {
						continue;
					}
					sum += combined_weight;

					double offset_x =
					    k + (val2 / combined_weight) - (width / 2);
					double offset_y = j - (height / 2);
					pc += snprintf(
					    pc, body_len - (ulong)(pc - shader_body),
					    shader_add, combined_weight, offset_x, offset_y);
					assert(pc < shader_body + body_len);
				}
			}
		} else {
			// use interpolation in y dimension (height)
			for (int j = 0; j < height; j += 2) {
				for (int k = 0; k < width; ++k) {
					double val1, val2;
					val1 = kern->data[j * width + k];
					val2 = (j + 1 < height)
					           ? kern->data[(j + 1) * width + k]
					           : 0;

					double combined_weight = val1 + val2;
					if (combined_weight == 0) {
						continue;
					}
					sum += combined_weight;

					double offset_x = k - (width / 2);
					double offset_y =
					    j + (val2 / combined_weight) - (height / 2);
					pc += snprintf(
					    pc, body_len - (ulong)(pc - shader_body),
					    shader_add, combined_weight, offset_x, offset_y);
					assert(pc < shader_body + body_len);
				}
			}
		}

		auto pass = ctx->blur_shader + i;
		size_t shader_len = strlen(FRAG_SHADER_BLUR) + strlen(shader_body) +
		                    10 /* sum */ + 1 /* null terminator */;
		char *shader_str = ccalloc(shader_len, char);
		auto real_shader_len =
		    snprintf(shader_str, shader_len, FRAG_SHADER_BLUR, shader_body, sum);
		CHECK(real_shader_len >= 0);
		CHECK((size_t)real_shader_len < shader_len);
		free(shader_body);

		// Build program
		pass->prog = gl_create_program_from_strv(
		    (const char *[]){vertex_shader, NULL},
		    (const char *[]){shader_str, scaled_masking_glsl, perturb_glsl, NULL});
		free(shader_str);
		if (!pass->prog) {
			log_error("Failed to create GLSL program.");
			success = false;
			goto out;
		}
		glBindFragDataLocation(pass->prog, 0, "out_color");

		// Setup projection matrix
		glUseProgram(pass->prog);
		glUniformMatrix4fv(UNIFORM_PROJECTION_LOC, 1, false, projection);
		glUseProgram(0);

		ctx->resize_width += kern->w / 2;
		ctx->resize_height += kern->h / 2;
	}

	if (nkernels == 1) {
		// Generate an extra null pass so we don't need special code path for
		// the single pass case
		auto pass = &ctx->blur_shader[1];
		pass->prog = gl_create_program_from_strv(
		    (const char *[]){vertex_shader, NULL},
		    (const char *[]){blend_with_mask_frag, scaled_masking_glsl,
		                     perturb_glsl, NULL});

		// Setup projection matrix
		glUseProgram(pass->prog);
		glUniformMatrix4fv(UNIFORM_PROJECTION_LOC, 1, false, projection);
		glUseProgram(0);

		ctx->npasses = 2;
	} else {
		ctx->npasses = nkernels;
	}

	success = true;
out:
	if (method != BLUR_METHOD_KERNEL) {
		// We generated the blur kernels, so we need to free them
		for (int i = 0; i < nkernels; i++) {
			free(kernels[i]);
		}
		free(kernels);
	}

	// Restore LC_NUMERIC
	setlocale(LC_NUMERIC, lc_numeric_old);
	free(lc_numeric_old);

	return success;
}

bool gl_create_dual_kawase_blur_context(void *blur_context, GLfloat *projection,
                                        enum blur_method method, void *args) {
	bool success = false;
	auto ctx = (struct gl_blur_context *)blur_context;

	ctx->method = method;

	auto blur_params = generate_dual_kawase_params(args);
	ctx->dual_kawase_strength = ((struct dual_kawase_blur_args *)args)->strength;

	// Specify required textures and FBOs
	ctx->blur_texture_count = blur_params->iterations;
	ctx->blur_fbo_count = blur_params->iterations;

	ctx->resize_width += blur_params->expand;
	ctx->resize_height += blur_params->expand;

	ctx->npasses = 2;
	ctx->blur_shader = ccalloc(ctx->npasses, struct gl_shader);

	char *lc_numeric_old = strdup(setlocale(LC_NUMERIC, NULL));
	// Enforce LC_NUMERIC locale "C" here to make sure decimal point is sane
	// Thanks to hiciu for reporting.
	setlocale(LC_NUMERIC, "C");

	// Dual-kawase downsample shader / program
	auto down_pass = ctx->blur_shader;
	{
		// clang-format off
		static const char *FRAG_SHADER_DOWN = GLSL(330,
			layout(location = UNIFORM_TEX_SRC_LOC)
			uniform sampler2D tex_src;
			layout(location = UNIFORM_SCALE_LOC)
			uniform float scale = 1.0;
			layout(location = UNIFORM_DUAL_KAWASE_SCALE_LOC)
			uniform float dual_kawase_scale = 1.0;
			layout(location = UNIFORM_PIXEL_NORM_LOC)
			uniform vec2 pixel_norm;
			in vec2 texcoord;
			out vec4 out_color;
			void main() {
				vec2 offset = %.7g * dual_kawase_scale * pixel_norm;
				vec2 uv = texcoord * pixel_norm * (2.0 / scale);
				vec4 sum = texture2D(tex_src, uv) * 4.0;
				sum += texture2D(tex_src, uv - vec2(0.5, 0.5) * offset);
				sum += texture2D(tex_src, uv + vec2(0.5, 0.5) * offset);
				sum += texture2D(tex_src, uv + vec2(0.5, -0.5) * offset);
				sum += texture2D(tex_src, uv - vec2(0.5, -0.5) * offset);
				out_color = sum / 8.0;
			}
		);
		// clang-format on

		// Build shader
		size_t shader_len =
		    strlen(FRAG_SHADER_DOWN) + 10 /* offset */ + 1 /* null terminator */;
		char *shader_str = ccalloc(shader_len, char);
		auto real_shader_len =
		    snprintf(shader_str, shader_len, FRAG_SHADER_DOWN, blur_params->offset);
		CHECK(real_shader_len >= 0);
		CHECK((size_t)real_shader_len < shader_len);

		// Build program
		down_pass->prog = gl_create_program_from_str(vertex_shader, shader_str);
		free(shader_str);
		if (!down_pass->prog) {
			log_error("Failed to create GLSL program.");
			success = false;
			goto out;
		}
		glBindFragDataLocation(down_pass->prog, 0, "out_color");

		// Setup projection matrix
		glUseProgram(down_pass->prog);
		glUniformMatrix4fv(UNIFORM_PROJECTION_LOC, 1, false, projection);
		glUseProgram(0);
	}

	// Dual-kawase upsample shader / program
	auto up_pass = ctx->blur_shader + 1;
	{
		// clang-format off
		static const char *FRAG_SHADER_UP = GLSL(330,
			layout(location = UNIFORM_TEX_SRC_LOC)
			uniform sampler2D tex_src;
			layout(location = UNIFORM_SCALE_LOC)
			uniform float scale = 1.0;
			layout(location = UNIFORM_DUAL_KAWASE_SCALE_LOC)
			uniform float dual_kawase_scale = 1.0;
			layout(location = UNIFORM_LIQUID_GLASS_SOURCE_SCALE_LOC)
			uniform float liquid_glass_source_scale = 0.5;
			layout(location = UNIFORM_PIXEL_NORM_LOC)
			uniform vec2 pixel_norm;
			layout(location = UNIFORM_OPACITY_LOC)
			uniform float opacity;
			layout(location = UNIFORM_BLUR_NOISE_RADIUS_LOC)
			uniform int noise_radius;
			layout(location = UNIFORM_BLUR_NOISE_SCALE_LOC)
			uniform float noise_scale;
			layout(location = UNIFORM_LIQUID_GLASS_STRENGTH_LOC)
			uniform int liquid_glass_strength;
			layout(location = UNIFORM_LIQUID_GLASS_ABERRATION_LOC)
			uniform float liquid_glass_aberration;
			layout(location = UNIFORM_LIQUID_GLASS_SATURATION_LOC)
			uniform float liquid_glass_saturation;
			layout(location = UNIFORM_LIQUID_GLASS_DISTANCE_LOC)
			uniform int liquid_glass_distance;
			layout(location = UNIFORM_LIQUID_GLASS_ZOOM_LOC)
			uniform float liquid_glass_zoom;
			layout(location = UNIFORM_LIQUID_GLASS_ANCHOR_LOC)
			uniform vec2 liquid_glass_anchor;
			layout(location = UNIFORM_LIQUID_GLASS_ANCHOR_MOUSE_LOC)
			uniform int liquid_glass_zoom_anchor_mouse;
			layout(location = UNIFORM_BLUR_GRADIENT_POINTS_LOC)
			uniform vec2 blur_gradient_points[32];
			layout(location = UNIFORM_BLUR_GRADIENT_START_LOC)
			uniform vec2 blur_gradient_start;
			layout(location = UNIFORM_BLUR_GRADIENT_END_LOC)
			uniform vec2 blur_gradient_end;
			layout(location = UNIFORM_BLUR_GRADIENT_POINT_COUNT_LOC)
			uniform int blur_gradient_point_count;
			layout(location = UNIFORM_BLUR_GRADIENT_ENABLED_LOC)
			uniform int blur_gradient_enabled;
			layout(location = UNIFORM_LIQUID_GLASS_SOURCE_TEX_LOC)
			uniform sampler2D liquid_glass_source_tex;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_COUNT_LOC)
			uniform int blur_gradient_level_count;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_0_LOC)
			uniform sampler2D blur_gradient_level_0;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_1_LOC)
			uniform sampler2D blur_gradient_level_1;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_2_LOC)
			uniform sampler2D blur_gradient_level_2;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_3_LOC)
			uniform sampler2D blur_gradient_level_3;
			layout(location = UNIFORM_BLUR_GRADIENT_LEVEL_4_LOC)
			uniform sampler2D blur_gradient_level_4;
			layout(location = UNIFORM_LIQUID_GLASS_CURVE_X1_LOC)
			uniform float liquid_glass_curve_x1;
			layout(location = UNIFORM_LIQUID_GLASS_CURVE_Y1_LOC)
			uniform float liquid_glass_curve_y1;
			layout(location = UNIFORM_LIQUID_GLASS_CURVE_X2_LOC)
			uniform float liquid_glass_curve_x2;
			layout(location = UNIFORM_LIQUID_GLASS_CURVE_Y2_LOC)
			uniform float liquid_glass_curve_y2;
			// These declarations are also present in scaled_masking_glsl. They
			// must be repeated here because each fragment shader is compiled
			// independently before the program is linked.
			layout(location = UNIFORM_MASK_TEX_LOC)
			uniform sampler2D mask_tex;
			layout(location = UNIFORM_MASK_OFFSET_LOC)
			uniform vec2 mask_offset;
			layout(location = UNIFORM_MASK_CORNER_RADIUS_LOC)
			uniform float mask_corner_radius;
			layout(location = UNIFORM_MASK_SCALE_LOC)
			uniform vec2 mask_scale;
			in vec2 texcoord;
			out vec4 out_color;
			float mask_factor();
			vec2 perturb(vec2, float, float);
			float blur_gradient_amount() {
				if (blur_gradient_point_count <= 0) return 1.0;
				vec2 gradient = blur_gradient_end - blur_gradient_start;
				float gradient_length_squared = dot(gradient, gradient);
				float position = gradient_length_squared > 1e-8
				    ? clamp(dot(texcoord - blur_gradient_start, gradient) / gradient_length_squared, 0.0, 1.0)
				    : 0.0;
				vec2 previous = blur_gradient_points[0];
				if (position <= previous.x) return previous.y;
				for (int i = 1; i < 32; ++i) {
					if (i >= blur_gradient_point_count) break;
					vec2 next = blur_gradient_points[i];
					if (position <= next.x) {
						return mix(previous.y, next.y,
						           (position - previous.x) / (next.x - previous.x));
					}
					previous = next;
				}
				return previous.y;
			}
			vec4 blur_gradient_sample_level(int level, vec2 sample_uv) {
				if (level <= 0) return texture2D(liquid_glass_source_tex, sample_uv);
				if (level == 1) return texture2D(blur_gradient_level_0, sample_uv);
				if (level == 2) return texture2D(blur_gradient_level_1, sample_uv);
				if (level == 3) return texture2D(blur_gradient_level_2, sample_uv);
				if (level == 4) return texture2D(blur_gradient_level_3, sample_uv);
				return texture2D(blur_gradient_level_4, sample_uv);
			}
			vec4 blur_gradient_sample(vec2 sample_uv) {
				float amount = blur_gradient_amount();
				float level = amount * float(blur_gradient_level_count);
				int low_level = int(floor(level));
				int high_level = min(low_level + 1, blur_gradient_level_count);
				return mix(blur_gradient_sample_level(low_level, sample_uv),
				           blur_gradient_sample_level(high_level, sample_uv), fract(level));
			}
			vec4 liquid_glass_blur(vec2 sample_uv, vec2 sample_offset) {
				vec4 result = blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(-1.0, 0.0) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(-1.0, 0.0) * sample_offset);
				result += (blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(-0.5, 0.5) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(-0.5, 0.5) * sample_offset)) * 2.0;
				result += blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(0.0, 1.0) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(0.0, 1.0) * sample_offset);
				result += (blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(0.5, 0.5) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(0.5, 0.5) * sample_offset)) * 2.0;
				result += blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(1.0, 0.0) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(1.0, 0.0) * sample_offset);
				result += (blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(0.5, -0.5) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(0.5, -0.5) * sample_offset)) * 2.0;
				result += blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(0.0, -1.0) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(0.0, -1.0) * sample_offset);
				result += (blur_gradient_enabled != 0
				    ? blur_gradient_sample(sample_uv + vec2(-0.5, -0.5) * sample_offset)
				    : texture2D(tex_src, sample_uv + vec2(-0.5, -0.5) * sample_offset)) * 2.0;
				return result / 12.0;
			}
			float liquid_glass_bezier(float x) {
				// cubic-bezier control points, solved from x to its parameter.
				float t = x;
				for (int i = 0; i < 5; ++i) {
					float omt = 1.0 - t;
					float bx = 3.0 * omt * omt * t * liquid_glass_curve_x1 +
					           3.0 * omt * t * t * liquid_glass_curve_x2 + t * t * t;
					float dx = 3.0 * omt * omt * liquid_glass_curve_x1 +
					           6.0 * omt * t * (liquid_glass_curve_x2 - liquid_glass_curve_x1) +
					           3.0 * t * t * (1.0 - liquid_glass_curve_x2);
					t -= (bx - x) / max(dx, 1e-4);
					t = clamp(t, 0.0, 1.0);
				}
				float omt = 1.0 - t;
				return 3.0 * omt * omt * t * liquid_glass_curve_y1 +
				       3.0 * omt * t * t * liquid_glass_curve_y2 + t * t * t;
			}
			void main() {
				float gradient_amount = blur_gradient_enabled != 0 ? blur_gradient_amount() : 1.0;
				vec2 offset = %.7g * dual_kawase_scale * pixel_norm * gradient_amount;
				vec2 uv = perturb(texcoord, noise_radius, noise_scale) * pixel_norm * liquid_glass_source_scale / scale;
				%s
				vec4 base_color = liquid_glass_blur(uv, offset);
				vec4 red_color = liquid_glass_blur(uv + aberration_offset, offset);
				vec4 blue_color = liquid_glass_blur(uv - aberration_offset, offset);
				vec4 color = vec4(red_color.r, base_color.g, blue_color.b, base_color.a);
				float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
				color.rgb = mix(vec3(luminance), color.rgb, liquid_glass_saturation);
				out_color = color * opacity * mask_factor();
			}
		);
		// clang-format on

		// Build shader
		const char *warp = method == BLUR_METHOD_LIQUID_GLASS
                               ? "vec2 mask_size = textureSize(mask_tex, 0);\n"
                                  "vec2 maskcoord = (texcoord - mask_offset) / mask_scale;\n"
                                  "vec2 centered = maskcoord - mask_size * 0.5;\n"
                                  "vec2 half_size = mask_size * 0.5;\n"
                                  "float edge_width = float(liquid_glass_distance);\n"

                                  "vec2 window_center = (mask_offset + mask_size * 0.5) * pixel_norm * liquid_glass_source_scale / scale;\n"
                                  "vec2 zoom_origin = window_center;\n"
                                  "if (liquid_glass_zoom_anchor_mouse == 1) {\n"
                                  "    zoom_origin = liquid_glass_anchor * pixel_norm * liquid_glass_source_scale / scale;\n"
                                  "}\n"
                                   "uv = mix(zoom_origin, uv, liquid_glass_zoom);\n"

                                  // Build a normalized radial field over the complete
                                  // window. Unlike the old edge-band calculation, this
                                  // has no rectangular dead zone in the middle: every
                                  // pixel except the exact center receives a smooth
                                  // fisheye displacement when distance covers the window.
                                  "vec2 normalized_centered = centered / max(half_size, vec2(1.0));\n"
                                  // Use picom's rounded-corner distance to shape the
                                  // fisheye falloff without a separate corner term.
                                  "float warp_corner_radius = mask_corner_radius;\n"
                                  "vec2 rounded_half_size = max(half_size - vec2(warp_corner_radius), vec2(0.0));\n"
                                  "vec2 rounded_q = abs(centered) - rounded_half_size;\n"
                                  "float rounded_sdf = length(max(rounded_q, vec2(0.0))) + min(max(rounded_q.x, rounded_q.y), 0.0) - warp_corner_radius;\n"
                                  "float distance_to_edge = max(-rounded_sdf, 0.0);\n"
                                   "float edge_progress = edge_width > 0.0 ? 1.0 - smoothstep(0.0, edge_width, distance_to_edge) : 0.0;\n"
                                   "edge_progress = liquid_glass_bezier(clamp(edge_progress, 0.0, 1.0));\n"
                                  // Apply one radial fisheye displacement. The rounded
                                  // SDF still controls the edge falloff, but there is
                                  // no separate corner displacement term.
                                  "vec2 radial_direction = normalized_centered / max(length(normalized_centered), 1e-4);\n"
                                    "vec2 stretch = -radial_direction * edge_progress;\n"

                                   "uv += stretch * float(liquid_glass_strength) * 3.6\n"
                                  "    * pixel_norm / (2.0 * scale);\n"
                                  "vec2 aberration_offset = stretch * liquid_glass_aberration\n"
                                  "    * pixel_norm / (2.0 * scale);\n"
                                   : "vec2 aberration_offset = vec2(0.0);\n";
		size_t shader_len = strlen(FRAG_SHADER_UP) + strlen(warp) + 24 + 1;
		char *shader_str = ccalloc(shader_len, char);
		auto real_shader_len =
		    snprintf(shader_str, shader_len, FRAG_SHADER_UP, blur_params->offset, warp);
		CHECK(real_shader_len >= 0);
		CHECK((size_t)real_shader_len < shader_len);

		// Build program
		up_pass->prog = gl_create_program_from_strv(
		    (const char *[]){vertex_shader, NULL},
		    (const char *[]){shader_str, scaled_masking_glsl, perturb_glsl, NULL});
		free(shader_str);
		if (!up_pass->prog) {
			log_error("Failed to create GLSL program.");
			success = false;
			goto out;
		}
		glBindFragDataLocation(up_pass->prog, 0, "out_color");

		// Setup projection matrix
		glUseProgram(up_pass->prog);
		glUniformMatrix4fv(UNIFORM_PROJECTION_LOC, 1, false, projection);
		glUseProgram(0);
	}

	success = true;
out:
	free(blur_params);

	// Restore LC_NUMERIC
	setlocale(LC_NUMERIC, lc_numeric_old);
	free(lc_numeric_old);

	return success;
}

void *gl_create_blur_context(backend_t *base, enum blur_method method,
                             enum backend_image_format format, struct blur_args *args) {
	bool success;
	auto gd = (struct gl_data *)base;

	auto ctx = ccalloc(1, struct gl_blur_context);

	if (!method || method >= BLUR_METHOD_INVALID) {
		ctx->method = BLUR_METHOD_NONE;
		return ctx;
	}

	// Set projection matrix to gl viewport dimensions so we can use screen
	// coordinates for all vertices
	// Note: OpenGL matrices are column major
	GLint viewport_dimensions[2];
	glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewport_dimensions);
	GLfloat projection_matrix[4][4] = {{2.0F / (GLfloat)viewport_dimensions[0], 0, 0, 0},
	                                   {0, 2.0F / (GLfloat)viewport_dimensions[1], 0, 0},
	                                   {0, 0, 0, 0},
	                                   {-1, -1, 0, 1}};

	if (method == BLUR_METHOD_DUAL_KAWASE || method == BLUR_METHOD_LIQUID_GLASS) {
		success = gl_create_dual_kawase_blur_context(ctx, projection_matrix[0],
		                                             method, args);
	} else {
		success =
		    gl_create_kernel_blur_context(ctx, projection_matrix[0], method, args);
	}
	if (!success || ctx->method == BLUR_METHOD_NONE) {
		goto out;
	}

	ctx->noise_radius = args->noise_radius;
	ctx->noise_scale = (float)args->noise_scale;
	if (method == BLUR_METHOD_LIQUID_GLASS) {
		auto glass_args = (struct dual_kawase_blur_args *)args;
		ctx->liquid_glass_strength = glass_args->liquid_glass_strength;
		ctx->liquid_glass_distance = glass_args->liquid_glass_distance;
		for (int i = 0; i < 4; i++) {
			ctx->liquid_glass_curve[i] = (float)glass_args->liquid_glass_curve[i];
		}
        // Keep enough source texture around the window for the stronger warp.
        // Without this, the shader samples outside the resized blur region and
        // the fisheye appears to stop abruptly, especially near diagonals.
        const int warp_resize = (int)ceil(ctx->liquid_glass_strength * 3.6);
        ctx->resize_width = max2(ctx->resize_width, warp_resize);
        ctx->resize_height = max2(ctx->resize_height, warp_resize);
	}
	// Expand the blur region to account for noise samples
	ctx->resize_width = max2(ctx->resize_width, ctx->noise_radius);
	ctx->resize_height = max2(ctx->resize_height, ctx->noise_radius);

	// Texture size will be defined by gl_blur
	ctx->blur_textures = ccalloc(ctx->blur_texture_count, GLuint);
	ctx->texture_sizes = ccalloc(ctx->blur_texture_count, struct texture_size);
	ctx->format = format;
	glGenTextures(ctx->blur_texture_count, ctx->blur_textures);

	// Generate FBO and textures when needed
	ctx->blur_fbos = ccalloc(ctx->blur_fbo_count, GLuint);
	glGenFramebuffers(ctx->blur_fbo_count, ctx->blur_fbos);

	for (int i = 0; i < ctx->blur_fbo_count; ++i) {
		if (!ctx->blur_fbos[i]) {
			log_error("Failed to generate framebuffer objects for blur");
			success = false;
			goto out;
		}
	}

out:
	if (!success) {
		gl_destroy_blur_context(&gd->base, ctx);
		ctx = NULL;
	}

	gl_check_err();
	return ctx;
}

void gl_get_blur_size(void *blur_context, int *width, int *height) {
	auto ctx = (struct gl_blur_context *)blur_context;
	*width = ctx->resize_width;
	*height = ctx->resize_height;
}
