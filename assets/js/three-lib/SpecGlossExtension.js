// Minimal KHR_materials_pbrSpecularGlossiness support for GLTFLoader.
// Core three.js dropped this legacy extension years ago; some Sketchfab
// exports still require it. This approximates it onto MeshStandardMaterial
// (diffuse -> color/map, glossiness -> inverse roughness) instead of a full
// spec/gloss shader, which is enough to recover correct color and texture.
import { Color, MeshStandardMaterial, SRGBColorSpace, LinearSRGBColorSpace } from './three.module.min.js';

export class GLTFMaterialsPbrSpecularGlossinessExtension {

	constructor( parser ) {

		this.parser = parser;
		this.name = 'KHR_materials_pbrSpecularGlossiness';

	}

	getMaterialType( materialIndex ) {

		const parser = this.parser;
		const materialDef = parser.json.materials[ materialIndex ];

		if ( ! materialDef.extensions || ! materialDef.extensions[ this.name ] ) return null;

		return MeshStandardMaterial;

	}

	extendMaterialParams( materialIndex, materialParams ) {

		const parser = this.parser;
		const materialDef = parser.json.materials[ materialIndex ];

		if ( ! materialDef.extensions || ! materialDef.extensions[ this.name ] ) {

			return Promise.resolve();

		}

		const pending = [];
		const extension = materialDef.extensions[ this.name ];

		materialParams.color = new Color( 1, 1, 1 );
		materialParams.opacity = 1;

		if ( Array.isArray( extension.diffuseFactor ) ) {

			const array = extension.diffuseFactor;
			materialParams.color.setRGB( array[ 0 ], array[ 1 ], array[ 2 ], LinearSRGBColorSpace );
			materialParams.opacity = array[ 3 ] !== undefined ? array[ 3 ] : 1;

		}

		if ( extension.diffuseTexture !== undefined ) {

			pending.push( parser.assignTexture( materialParams, 'map', extension.diffuseTexture, SRGBColorSpace ) );

		}

		materialParams.emissive = new Color( 0, 0, 0 );
		materialParams.roughness = extension.glossinessFactor !== undefined ? 1 - extension.glossinessFactor : 0.5;
		materialParams.metalness = 0.15;

		if ( extension.specularGlossinessTexture !== undefined ) {

			pending.push( parser.assignTexture( materialParams, 'roughnessMap', extension.specularGlossinessTexture ) );

		}

		return Promise.all( pending );

	}

}
