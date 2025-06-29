//! Test file to verify fusion implementations

#[cfg(test)]
mod tests {
    use crate::models::architectures::fusion::*;
    use crate::layers::Layer;
    use ndarray::Array;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_feature_alignment_backward_update() -> crate::error::Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut alignment: FeatureAlignment<f32> = FeatureAlignment::new(10, 8, Some("test"))?;
        
        // Test forward pass
        let input = Array::ones((2, 10)).into_dyn();
        let output = alignment.forward(&input)?;
        assert_eq!(output.shape(), &[2, 8]);
        
        // Test backward pass
        let grad_output = Array::ones((2, 8)).into_dyn();
        let grad_input = alignment.backward(&input, &grad_output)?;
        assert_eq!(grad_input.shape(), input.shape());
        
        // Test update
        alignment.update(0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_cross_modal_attention_backward_update() -> crate::error::Result<()> {
        let mut attention: CrossModalAttention<f32> = CrossModalAttention::new(8, 8, 8)?;
        
        // Test dedicated forward method
        let query = Array::ones((2, 4, 8)).into_dyn();
        let context = Array::ones((2, 6, 8)).into_dyn();
        let output = attention.forward(&query, &context)?;
        assert_eq!(output.shape(), &[2, 4, 8]);
        
        // Test Layer trait methods (simplified)
        let dummy_input = Array::ones((2, 4, 8)).into_dyn();
        let grad_output = Array::ones((2, 4, 8)).into_dyn();
        let grad_input = attention.backward(&dummy_input, &grad_output)?;
        assert_eq!(grad_input.shape(), grad_output.shape());
        
        // Test update
        attention.update(0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_film_module_backward_update() -> crate::error::Result<()> {
        let mut film: FiLMModule<f32> = FiLMModule::new(8, 6)?;
        
        // Test dedicated forward method
        let features = Array::ones((2, 8)).into_dyn();
        let conditioning = Array::ones((2, 6)).into_dyn();
        let output = film.forward(&features, &conditioning)?;
        assert_eq!(output.shape(), &[2, 8]);
        
        // Test Layer trait methods (simplified)
        let dummy_input = Array::ones((2, 8)).into_dyn();
        let grad_output = Array::ones((2, 8)).into_dyn();
        let grad_input = film.backward(&dummy_input, &grad_output)?;
        assert_eq!(grad_input.shape(), grad_output.shape());
        
        // Test update
        film.update(0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_bilinear_fusion_backward_update() -> crate::error::Result<()> {
        let mut bilinear: BilinearFusion<f32> = BilinearFusion::new(8, 6, 10, 4)?;
        
        // Test dedicated forward method
        let features_a = Array::ones((2, 8)).into_dyn();
        let features_b = Array::ones((2, 6)).into_dyn();
        let output = bilinear.forward(&features_a, &features_b)?;
        assert_eq!(output.shape(), &[2, 10]);
        
        // Test Layer trait methods (simplified)
        let dummy_input = Array::ones((2, 8)).into_dyn();
        let grad_output = Array::ones((2, 10)).into_dyn();
        let grad_input = bilinear.backward(&dummy_input, &grad_output)?;
        assert_eq!(grad_input.shape(), grad_output.shape());
        
        // Test update
        bilinear.update(0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_feature_fusion_backward_update() -> crate::error::Result<()> {
        let config = FeatureFusionConfig {
            input_dims: vec![10, 8],
            hidden_dim: 6,
            fusion_method: FusionMethod::Concatenation,
            dropout_rate: 0.1,
            num_classes: 3,
            include_head: true,
        };
        
        let mut fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        
        // Test forward_multi
        let inputs = vec![
            Array::ones((2, 10)).into_dyn(),
            Array::ones((2, 8)).into_dyn(),
        ];
        let output = fusion.forward_multi(&inputs)?;
        assert_eq!(output.shape(), &[2, 3]);
        
        // Test Layer trait methods (simplified)
        let dummy_input = Array::ones((2, 10)).into_dyn();
        let grad_output = Array::ones((2, 3)).into_dyn();
        let grad_input = fusion.backward(&dummy_input, &grad_output)?;
        assert_eq!(grad_input.shape(), grad_output.shape());
        
        // Test update
        fusion.update(0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_attention_fusion() -> crate::error::Result<()> {
        let config = FeatureFusionConfig {
            input_dims: vec![10, 8],
            hidden_dim: 6,
            fusion_method: FusionMethod::Attention,
            dropout_rate: 0.1,
            num_classes: 3,
            include_head: true,
        };
        
        let fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        
        // Test forward_multi with attention fusion
        let inputs = vec![
            Array::ones((2, 10)).into_dyn(),
            Array::ones((2, 8)).into_dyn(),
        ];
        let output = fusion.forward_multi(&inputs)?;
        assert_eq!(output.shape(), &[2, 3]);
        
        Ok(())
    }

    #[test]
    fn test_film_fusion() -> crate::error::Result<()> {
        let config = FeatureFusionConfig {
            input_dims: vec![10, 8],
            hidden_dim: 6,
            fusion_method: FusionMethod::FiLM,
            dropout_rate: 0.1,
            num_classes: 3,
            include_head: true,
        };
        
        let fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        
        // Test forward_multi with FiLM fusion
        let inputs = vec![
            Array::ones((2, 10)).into_dyn(),
            Array::ones((2, 8)).into_dyn(),
        ];
        let output = fusion.forward_multi(&inputs)?;
        assert_eq!(output.shape(), &[2, 3]);
        
        Ok(())
    }

    #[test]
    fn test_bilinear_fusion_model() -> crate::error::Result<()> {
        let config = FeatureFusionConfig {
            input_dims: vec![10, 8],
            hidden_dim: 6,
            fusion_method: FusionMethod::Bilinear,
            dropout_rate: 0.1,
            num_classes: 3,
            include_head: true,
        };
        
        let fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        
        // Test forward_multi with bilinear fusion
        let inputs = vec![
            Array::ones((2, 10)).into_dyn(),
            Array::ones((2, 8)).into_dyn(),
        ];
        let output = fusion.forward_multi(&inputs)?;
        assert_eq!(output.shape(), &[2, 3]);
        
        Ok(())
    }
}