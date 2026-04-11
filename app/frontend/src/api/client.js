/**
 * Universal API client for communicating with the Flask backend.
 * 
 * Automatically handles:
 * - JSON serialization for non-FormData bodies
 * - Content-Type headers
 * - HTTP error detection (throws on non-2xx responses)
 */
export async function callApi(endpoint, method = 'GET', body = null) {
	const options = { method };

	if (body) {
		if (body instanceof FormData) {
			options.body = body;
			// Let browser set Content-Type with boundary for FormData
		} else {
			options.headers = { 'Content-Type': 'application/json' };
			options.body = JSON.stringify(body);
		}
	}

	const res = await fetch(`/api/${endpoint}`, options);
	const data = await res.json().catch(() => ({}));

	if (!res.ok) {
		const message = data.error || data.details || `HTTP ${res.status}`;
		throw new Error(message);
	}

	return data;
}