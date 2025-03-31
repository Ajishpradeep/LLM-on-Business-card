import gradio as gr
from dotenv import load_dotenv
from src.search_engine import BusinessCardVectorDB
from src.business_card_processor import BusinessCardProcessor

# Load environment variables
load_dotenv()


def format_extracted_info_for_display(extracted_info):
    """Format the extracted information for better display in Gradio."""
    html_output = "<div style='font-family: Arial; padding: 10px;'>"

    # Primary Info
    if "primary_info" in extracted_info:
        primary = extracted_info["primary_info"]
        html_output += "<h3>Primary Information</h3>"
        html_output += f"<p><strong>Name:</strong> {primary.get('name', {}).get('value', 'N/A')} (Confidence: {primary.get('name', {}).get('confidence', 'N/A')})</p>"
        html_output += f"<p><strong>Job Title:</strong> {primary.get('job_title', {}).get('value', 'N/A')} (Confidence: {primary.get('job_title', {}).get('confidence', 'N/A')})</p>"
        company = primary.get("company", {})
        html_output += (
            f"<p><strong>Company:</strong> {company.get('text_value', 'N/A')} "
        )
        html_output += (
            f"(Logo identified: {'Yes' if company.get('logo_identified') else 'No'}, "
        )
        html_output += (
            f"QR Code: {'Yes' if company.get('QRcode_identifies') else 'No'}, "
        )
        html_output += f"Confidence: {company.get('confidence', 'N/A')})</p>"

    # Contact Info
    if "contact_info" in extracted_info:
        contact = extracted_info["contact_info"]
        html_output += "<h3>Contact Information</h3>"

        if contact.get("emails"):
            html_output += "<p><strong>Emails:</strong></p><ul>"
            for email in contact["emails"]:
                html_output += f"<li>{email.get('value', 'N/A')} (Type: {email.get('type', 'N/A')}, Confidence: {email.get('confidence', 'N/A')})</li>"
            html_output += "</ul>"

        if contact.get("phones"):
            html_output += "<p><strong>Phone Numbers:</strong></p><ul>"
            for phone in contact["phones"]:
                html_output += f"<li>{phone.get('value', 'N/A')} (Type: {phone.get('type', 'N/A')}, Confidence: {phone.get('confidence', 'N/A')})</li>"
            html_output += "</ul>"

        if contact.get("addresses"):
            html_output += "<p><strong>Addresses:</strong></p><ul>"
            for addr in contact["addresses"]:
                html_output += f"<li>{addr.get('value', 'N/A')} (Type: {addr.get('type', 'N/A')}, Confidence: {addr.get('confidence', 'N/A')})</li>"
            html_output += "</ul>"

    # Digital Presence
    if "digital_presence" in extracted_info:
        digital = extracted_info["digital_presence"]
        html_output += "<h3>Digital Presence</h3>"

        if "website" in digital:
            html_output += f"<p><strong>Website:</strong> {digital['website'].get('value', 'N/A')} (Confidence: {digital['website'].get('confidence', 'N/A')})</p>"

        if digital.get("social_media"):
            html_output += "<p><strong>Social Media:</strong></p><ul>"
            for sm in digital["social_media"]:
                html_output += (
                    f"<li>{sm.get('platform', 'N/A')}: {sm.get('handle', 'N/A')} "
                )
                html_output += f"(Source: {sm.get('identified_from', 'N/A')}, Confidence: {sm.get('confidence', 'N/A')})</li>"
            html_output += "</ul>"

    # Contextual Summary
    if "contextual_summary" in extracted_info:
        context = extracted_info["contextual_summary"]
        html_output += "<h3>Contextual Analysis</h3>"
        html_output += f"<p><strong>Professional Summary:</strong> {context.get('professional_summary', 'N/A')}</p>"

    html_output += "</div>"
    return html_output


def add_business_card(image_file):
    """Process business card image and add to vector database."""
    if not image_file:
        yield "<p style='color: orange;'>Please upload an image.</p>", None
        return

    # Yield loading state
    yield (
        "<p>Processing image and extracting information... This might take a moment.</p>",
        None,
    )

    try:
        # Initialize processor and vector DB
        processor = BusinessCardProcessor()
        db = BusinessCardVectorDB()

        # Extract information from image
        yield "<p>Extracting data using AI model...</p>", None
        card_data = processor.extract_from_image(image_file)

        # Add to database and get card ID
        yield "<p>Adding information to the database...</p>", None
        card_id = db.add_business_card(card_data)

        # Get the complete card data
        complete_card = db.get_card_by_id(card_id)

        # Format output for display
        html_output = format_extracted_info_for_display(card_data["extracted_info"])

        # Yield final results
        yield html_output, complete_card

    except Exception as e:
        # Yield error message
        yield f"<p style='color: red;'>Error processing image: {str(e)}</p>", None


def search_business_cards(query: str, num_results: int = 5):
    """Search for business cards in the vector database."""
    try:
        # Initialize vector DB
        db = BusinessCardVectorDB()

        # Search for cards
        results = db.search_cards(query, num_results)

        if not results:
            return (
                "<div style='color: red; padding: 10px;'>No matching business cards found.</div>",
                None,
            )

        # Format results for display
        html_output = "<div style='font-family: Arial; padding: 10px;'>"
        html_output += f"<h2>Search Results for: {query}</h2>"
        html_output += f"<p>Found {len(results)} matching business cards</p>"

        for i, result in enumerate(results, 1):
            html_output += "<div style='margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px;'>"
            html_output += f"<h3>Match #{i}</h3>"

            # Display the formatted extracted info
            if "extracted_info" in result:
                html_output += format_extracted_info_for_display(
                    result["extracted_info"]
                )

            # Display relevance score
            relevance_score = 1 - result.get("distance", 0)
            html_output += f"<p><strong>Match Score:</strong> {relevance_score:.1%}</p>"

            html_output += "</div>"

        html_output += "</div>"
        return html_output, results

    except Exception as e:
        error_msg = f"Error searching business cards: {str(e)}"
        print(error_msg)
        return f"<div style='color: red; padding: 10px;'>{error_msg}</div>", None


def create_gradio_interface():
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="Business Card Manager", theme="soft") as demo:
        gr.Markdown("# Business Card Manager")
        gr.Markdown(
            "Add business cards to the database and search through them semantically."
        )

        with gr.Tabs():
            # Add Business Card Tab
            with gr.Tab("Add Business Card"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Business Card",
                            type="filepath",
                            sources=["upload"],
                            interactive=True,
                        )
                        add_btn = gr.Button("Add to Database", variant="primary")

                    with gr.Column():
                        add_output_html = gr.HTML(label="Extracted Information")
                        add_output_json = gr.JSON(
                            label="Raw JSON Output", visible=False
                        )

                # Example inputs for quick testing
                gr.Examples(
                    examples=[
                        ["input/ajish_bcard.png"],
                    ],
                    inputs=[image_input],
                    outputs=[add_output_html, add_output_json],
                    fn=add_business_card,
                    cache_examples=False,
                    label="Try Example Images",
                )

            # Search Business Cards Tab
            with gr.Tab("Search Business Cards"):
                with gr.Row():
                    with gr.Column():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter search terms (e.g., 'AI researcher in Taiwan')",
                            lines=2,
                        )
                        num_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Results",
                        )
                        search_btn = gr.Button("Search", variant="primary")

                    with gr.Column():
                        search_output_html = gr.HTML(label="Search Results")
                        search_output_json = gr.JSON(
                            label="Raw JSON Output", visible=False
                        )

                # Example searches
                gr.Examples(
                    examples=[
                        ["AI researcher in Taiwan"],
                        ["Software engineer with machine learning experience"],
                        ["Business development in technology sector"],
                    ],
                    inputs=[search_input],
                    outputs=[search_output_html, search_output_json],
                    fn=lambda q: search_business_cards(q),
                    cache_examples=False,
                    label="Try Example Searches",
                )

        # Set up event handlers
        add_btn.click(
            fn=add_business_card,
            inputs=[image_input],
            outputs=[add_output_html, add_output_json],
        )

        search_btn.click(
            fn=search_business_cards,
            inputs=[search_input, num_results],
            outputs=[search_output_html, search_output_json],
        )

    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
