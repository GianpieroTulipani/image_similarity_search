# Value Proposition
The primary end-users of this machine learning system are customers using the e-commerce platform who are interested in finding and purchasing products similar to those they see on the platform, either from product pages or curated total look pages. The system aims to provide a seamless and convenient shopping experience by allowing users to easily locate and purchase products that align with their visual inspiration. By offering a product-based visual search, users save time that would otherwise be spent manually searching through categories, while also receiving coherent product suggestions that match their specific interests. This personalized shopping journey is intended to enhance user satisfaction and increase the likelihood of purchase.

The user workflow begins with selecting images from product pages, total look pages allowing the system to analyze the selected. This images are then matched with similar items available on the platform and presented to the user as personalized shopping suggestions. Users can apply filters to narrow down results based on preferences such as brand, color, size, and price range, and they can view more information on product detail pages before proceeding to purchase.


# Prediction Task
The task is simply that of representation learning in a contrastive setting. In particular, the machine learning component of the system will produce a multi-modal embedding space which will be inhabited by high-level representations of the product images and of their textual descriptions.

The idea is that of exploiting the supervision of natural language in order to group together in the resulting embedding space products that not only share relevant visual features, but also other non-visual properties.

The model will then be fed an image for each of the products in the catalogue, in order to pre-compute their high-level representations, which will be then stored in a vector database.

# Decisions

Once the products that are most similar to the currently visualized item have been retrieved, the corresponding pages will be linked to the user in the page which is presently being displayed. Front pictures of the products will also be shown as a preview.

# Impact Simulation
A validation set of images with known product matches will be used to evaluate the model's accuracy and performance before full deployment.
Correct predictions will lead to increased user engagement, satisfaction, and higher conversion rates, which in turn leads to increased revenue for the platform. On the other hand, incorrect predictions could result in user frustration and a reduced level of trust in the platform's capabilities. 

# Making Predictions
The system makes real-time predictions immediately upon image selection, providing instant results to users


# Data Sources
The data used by the system includes product images and metadata from the Armani e-commerce catalogs, which serve as the primary source of information for training the models. Data includes product images, descriptions, categories, pricing, material, and style attributes.

External datasets, such as fashion image datasets like DeepFashion might be used for pre-training the model and enhancing robustness.


# Data Collection
The initial training set for the model is created using existing product images and metadata from the e-commerce catalog. To increase the diversity of training images, data augmentation techniques, such as rotation, scaling, and flipping, might be applied. The model is regularly retrained using new product images as the catalog updates, ensuring that the system remains up-to-date. The rate of data collection aligns with the rate of new product additions, and model updates are scheduled accordingly, typically on a weekly or bi-weekly basis.

# Features
The model input features are mainly images, textual descriptions. The system might also utilize product attributes, such as category, brand, color, and style.

# Building Models
Model updates are performed regularly, such as on a monthly basis, or whenever a performance drop is detected. The system is monitored for concept drift, and retraining is initiated as needed to maintain performance.

The model update process involves data preprocessing, training, validation, and deployment, which may require several hours to a full day depending on data size. Updates are scheduled during off-peak hours to minimize any negative impact on the platform's availability.

# Monitoring

The metrics used to quantify value creation include user engagement metrics, such as click-through rate (CTR) on recommended products and time spent viewing recommended items. Conversion metrics, such as add-to-cart rate and purchase rate, are also tracked to measure the impact on sales. Additionally, system performance metrics, such as prediction accuracy, response time, and system uptime, are monitored to ensure that the model is performing as expected.

User feedback is another critical component of monitoring, with user ratings and feedback collected to assess the usefulness of the results. Customer support inquiries related to the feature are also tracked to identify any issues or areas for improvement. Business impact measures, such as revenue growth, customer retention, and market differentiation, are used to evaluate the overall success of the system. The objective is to enhance user experience, drive sales, and maintain a competitive advantage in the market by offering an innovative search capability.
