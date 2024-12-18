# Dataset Card: Giorgio Armani E-commerce Dataset

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Domain Knowledge Overview](#domain-knowledge-overview)
3. [Folder Structure](#folder-structure)
4. [License and Permissions](#license-and-permissions)

## Dataset Description

This dataset contains proprietary data from the Giorgio Armani e-commerce platform, specifically focusing on product catalog and sales data for the seasons SS23, FW23, and SS24. It is structured to support project works for the course *Software Engineering for AI-Enabled Systems* at the University of Bari for the Academic Year 2024/2025.

### Key Points:
- **Purpose:** Strictly for academic use within the scope of the aforementioned course.
- **Usage Restrictions:** The dataset must not be shared publicly or outside the course participants.
- **Supplementary Documentation:** Refer to the `dataset_info.pdf` file for comprehensive details about the dataset structure and content.

## Domain Knowledge Overview

### Giorgio Armani Overview:
Giorgio Armani S.p.A is a leading luxury company that owns multiple brands, including Giorgio Armani, Emporio Armani (EA), Armani Exchange (AX), and EA7.

### Key Concepts in the Fashion Industry:
1. **Product Identification**:
   - Defined by a combination of three attributes:
     - **Model:** Specific design or style.
     - **Fabric:** Material used.
     - **Color:** Specific color of the product.
   - These attributes generate a unique identifier (MFC).

2. **Seasonality:**
   - Products are categorized by seasons (Spring/Summer - SS, Fall/Winter - FW).
   - Continuous products may span multiple seasons, while others are specific to a single season.

## Folder Structure

### 1. Images (`images/`)
Contains the `metadata.csv` file, which includes information about product images.

- **Columns:**
  - `mfc`: Unique identifier combining model, fabric, and color.
  - `model`: Style number of the product.
  - `fabric`: Fabric code.
  - `color`: Color code.
  - `season`: Season code (e.g., FW2023).
  - `view_label`: Perspective/angle of the image.
  - `content`: Type of image content (e.g., `ecommerce`, `swatch`, `sketch`, `other`).
  - `url`: Hosted URL for the image.
  - `path`: Local path in the dataset.

For details on `view_label`, refer to `ecommerce_views_guidelines.pdf`.

### 2. Descriptions (`descriptions/`)
Contains Excel files for product descriptions by brand:
- `AX_descriptions.xlsx`
- `EA_descriptions.xlsx`
- `GA_EA7_descriptions.xlsx`

**Details:**
- Each file contains sheets corresponding to different languages.
- Products differing only by color often share descriptions.
- The `inventory.csv` file complements this data with hierarchical information (e.g., categories, gender).

### 3. Sales (`sales/`)
File: `Germany_Sales_2023_EA.xlsx`

- Provides sales data for Emporio Armani products in Germany during 2023.
- Includes metrics such as product returns and net sales.

### 4. Products (`products/`)
File: `inventory.csv`

- Contains product hierarchy, gender, composition, and target age data.

## Notes on MFC Format

- **Image Data Format:** `0WGGG0IT_T0409_FBWF`
- **Other Data Format:** `0WGGG0ITT04091FBWF`

Differences in MFC format are due to technical reasons and can generally be ignored.

## License and Permissions

- This dataset is proprietary and shared strictly for educational purposes within the course.
- Unauthorized distribution or usage outside the course is strictly prohibited.
