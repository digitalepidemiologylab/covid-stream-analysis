def get_dtypes(usecols=None):
    """Gets dtypes for columns"""
    return {
            "id": str, 
            "text": str, 
            "in_reply_to_status_id": str,
            "in_reply_to_user_id": str,
            "quoted_status_id": str,
            "quoted_user_id": str,
            "retweeted_status_id": str,
            "retweeted_user_id": str,
            "created_at": str,
            "entities.user_mentions": str,
            "user.id": str,
            "user.screen_name": str,
            "user.name": str,
            "user.description": str,
            "user.timezone": str,
            "user.location": str,
            "user.num_followers": int,
            "user.num_following": int,
            "user.created_at": str, 
            "user.statuses_count": int,
            "user.is_verified": bool,
            "lang": str,
            "token_count": int,
            "is_retweet": bool,
            "has_quote": bool,
            "is_reply": bool,
            "matched_keywords": list,
            "longitude": float,
            "latitude": float,
            "country_code": str,
            "region": str,
            "subregion": str,
            "geo_type": int
            }
