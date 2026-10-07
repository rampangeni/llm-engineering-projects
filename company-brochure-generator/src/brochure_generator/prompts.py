"""Prompts for link selection and brochure writing."""

LINK_SYSTEM_PROMPT = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

BROCHURE_SYSTEM_PROMPT = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

HUMOROUS_BROCHURE_SYSTEM_PROMPT = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

TONES = {"professional": BROCHURE_SYSTEM_PROMPT, "humorous": HUMOROUS_BROCHURE_SYSTEM_PROMPT}


def links_user_prompt(url: str, links: list[str]) -> str:
    return (
        f"\nHere is the list of links on the website {url} -\n"
        "Please decide which of these are relevant web links for a brochure about the company, \n"
        "respond with the full https URL in JSON format.\n"
        "Do not include Terms of Service, Privacy, email links.\n\n"
        "Links (some might be relative links):\n\n" + "\n".join(links)
    )


def brochure_user_prompt_header(company_name: str) -> str:
    return (
        f"\nYou are looking at a company called: {company_name}\n"
        "Here are the contents of its landing page and other relevant pages;\n"
        "use this information to build a short brochure of the company in markdown without code blocks.\n\n"
    )
