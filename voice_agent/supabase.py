from voice_agent.config import get_settings
from supabase import AsyncClient, acreate_client
from supabase.client import ClientOptions

settings = get_settings()

# Create a supabase client for the user to respect auth policies
async def supabase_client(user_jwt: str) -> AsyncClient:
    return await acreate_client(
        settings.supabase_url,
        settings.supabase_key,
        options=ClientOptions(headers={
            'Authorization': f'Bearer {user_jwt}'
        })
    )